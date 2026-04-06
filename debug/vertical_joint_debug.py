"""
Debug visualization for vertical joint detection.

Operates on a small region (CURVE_DEBUG_CENTER_X +/- CURVE_DEBUG_RANGE).
Detects vertical joints using Nx+ and Nx- separately (left/right brick faces).
Produces joint lines, displacement (X deviation from base), and rotation plots
for each normal direction independently.

Run from repo root: python debug/vertical_joint_debug.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
import glob
from tqdm import tqdm

from config import FEET_TO_METERS

# ── Settings ────────────────────────────────────────────────────────────────

def _cfg(name, default):
    try:
        mod = sys.modules.get('config') or __import__('config')
        return getattr(mod, name, default)
    except Exception:
        return default

DEBUG_CENTER_X    = _cfg('CURVE_DEBUG_CENTER_X', None)
DEBUG_RANGE       = _cfg('CURVE_DEBUG_RANGE', 2.0)

NORMAL_KNN        = _cfg('JOINT_NORMAL_KNN', 30)
RASTER_RESOLUTION = _cfg('JOINT_RASTER_RESOLUTION', 0.01)
GAUSSIAN_SIGMA    = _cfg('JOINT_GAUSSIAN_SIGMA', 3.0)
PEAK_MIN_HEIGHT   = 0.01
PEAK_MIN_DIST_M   = 0.05
WINDOW_WIDTH      = 1.0
WINDOW_STEP       = 0.25

MATCH_TOLERANCE   = _cfg('JOINT_MATCH_TOLERANCE', 0.05)
SEARCH_DISTANCE   = 0.5   # meters — bridge staggered brick gaps
MIN_TRACK_PEAKS   = 15
SPLINE_SMOOTHING  = _cfg('JOINT_SPLINE_SMOOTHING', 1.0)

INFLUENCE_HALF_X  = 8 * FEET_TO_METERS / 12  # ±8 inches in meters

OUTPUT_DIR = "outputs/debug"
NORMALS_CACHE_DIR = "outputs/point_clouds/unrolled"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ── Normals ────────────────────────────────────────────────────────────────

def get_normals(points, wall_id):
    cache_path = os.path.join(NORMALS_CACHE_DIR, "normals_%s.ply" % wall_id)
    if os.path.exists(cache_path):
        print("  Loading cached normals from %s" % cache_path)
        pc = o3d.t.io.read_point_cloud(cache_path)
        cached_pts = pc.point.positions.numpy()
        if len(cached_pts) == len(points):
            return pc.point.normals.numpy()
        print("  Cache stale (%d vs %d points), recomputing" % (
            len(cached_pts), len(points)))

    print("  Computing normals (knn=%d)..." % NORMAL_KNN)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    normals = np.asarray(pcd.normals)

    os.makedirs(NORMALS_CACHE_DIR, exist_ok=True)
    pcd_t = o3d.t.geometry.PointCloud()
    pcd_t.point.positions = points.astype(np.float32)
    pcd_t.point.normals = normals.astype(np.float32)
    o3d.t.io.write_point_cloud(cache_path, pcd_t)
    print("  Cached normals to %s" % cache_path)
    return normals


# ── Rasterization ──────────────────────────────────────────────────────────

def rasterize_vertical(points, normals, resolution):
    """Rasterize Nx+ and Nx- separately."""
    x, z = points[:, 0], points[:, 2]
    nx = normals[:, 0]

    x_edges = np.arange(x.min(), x.max() + resolution, resolution)
    z_edges = np.arange(z.min(), z.max() + resolution, resolution)

    xi = np.clip(np.searchsorted(x_edges, x) - 1, 0, len(x_edges) - 2)
    zi = np.clip(np.searchsorted(z_edges, z) - 1, 0, len(z_edges) - 2)

    nc, nr = len(x_edges) - 1, len(z_edges) - 1
    flat = zi * nc + xi
    total = nr * nc

    nx_pos = np.clip(nx, 0, None)
    nx_neg = np.clip(-nx, 0, None)

    pos_sum = np.bincount(flat, weights=nx_pos, minlength=total)
    neg_sum = np.bincount(flat, weights=nx_neg, minlength=total)
    counts = np.bincount(flat, minlength=total)

    m = counts > 0
    raster_pos = np.zeros(total)
    raster_neg = np.zeros(total)
    raster_pos[m] = pos_sum[m] / counts[m]
    raster_neg[m] = neg_sum[m] / counts[m]

    return raster_pos.reshape(nr, nc), raster_neg.reshape(nr, nc), x_edges, z_edges


# ── Peak detection (windows slide along Z, peaks in X) ────────────────────

def detect_peaks_vertical(raster, x_edges, z_edges):
    """Slide windows along Z, find peaks in X profiles.

    Returns list of (window_center_z, peak_x_array).
    """
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    res = RASTER_RESOLUTION

    min_dist_bins = max(1, int(PEAK_MIN_DIST_M / res))
    win_rows = max(1, int(WINDOW_WIDTH / res))
    step_rows = max(1, int(WINDOW_STEP / res))

    detections = []
    row = 0
    while row < raster.shape[0]:
        re_ = min(row + win_rows, raster.shape[0])
        avg = np.mean(raster[row:re_, :], axis=0)
        cz = np.mean(z_c[row:re_])

        if len(avg) >= 3:
            sm = gaussian_filter1d(avg, sigma=GAUSSIAN_SIGMA)
            pi, _ = find_peaks(sm, height=PEAK_MIN_HEIGHT, distance=min_dist_bins)
            if len(pi) > 0:
                detections.append((cz, x_c[pi]))
        row += step_rows

    return detections


# ── Tracking with search distance ──────────────────────────────────────────

def link_peaks_to_tracks(detections):
    """Link peaks across windows with search distance for staggered gaps.

    Each detection is (window_z, peak_x_array). Tracks link X positions
    across adjacent Z-windows, allowing gaps up to SEARCH_DISTANCE.
    """
    if not detections:
        return []

    wc_vals = np.array([d[0] for d in detections])

    finished = []
    wc0, pp0 = detections[0]
    active = [{'w': [wc0], 'p': [pp0[i]], 'last_det': 0}
              for i in range(len(pp0))]

    for det_idx in range(1, len(detections)):
        wc, pp = detections[det_idx]
        used_p = set()

        if active and len(pp) > 0:
            eligible = [(ei, t) for ei, t in enumerate(active)
                        if abs(wc - wc_vals[t['last_det']]) <= SEARCH_DISTANCE + 1e-9]

            if eligible:
                elig_idx, elig_tracks = zip(*eligible)
                elig_pos = np.array([t['p'][-1] for t in elig_tracks])
                dists = np.abs(elig_pos[:, None] - pp[None, :])

                matched = set()
                for fi in np.argsort(dists.ravel()):
                    ei, pi = divmod(int(fi), len(pp))
                    if ei in matched or pi in used_p:
                        continue
                    if dists[ei, pi] > MATCH_TOLERANCE:
                        break
                    t = elig_tracks[ei]
                    t['w'].append(wc)
                    t['p'].append(pp[pi])
                    t['last_det'] = det_idx
                    matched.add(ei)
                    used_p.add(pi)

        # Expire or keep
        new_active = []
        for t in active:
            if abs(wc - wc_vals[t['last_det']]) > SEARCH_DISTANCE + 1e-9:
                finished.append(t)
            else:
                new_active.append(t)

        for pi in range(len(pp)):
            if pi not in used_p:
                new_active.append({'w': [wc], 'p': [pp[pi]], 'last_det': det_idx})

        active = new_active

    finished.extend(active)

    # Convert to (x, z) tracks — for vertical: w=z, p=x
    tracks = []
    for t in finished:
        if len(t['w']) >= MIN_TRACK_PEAKS:
            tracks.append(list(zip(t['p'], t['w'])))  # (x, z) pairs
    return tracks


# ── Spline fitting ─────────────────────────────────────────────────────────

def fit_tracks(tracks):
    fitted = []
    for track in tracks:
        x = np.array([p[0] for p in track])
        z = np.array([p[1] for p in track])
        entry = {
            'x': x, 'z': z,
            'x_min': x.min(), 'x_max': x.max(),
            'z_min': z.min(), 'z_max': z.max(),
            'mean_x': x.mean(), 'mean_z': z.mean(),
            'tck': None,
        }
        if len(z) >= 4:
            try:
                tck, _ = splprep([x, z], s=SPLINE_SMOOTHING)
                entry['tck'] = tck
            except Exception:
                pass
        fitted.append(entry)

    fitted.sort(key=lambda t: t['mean_x'])
    return fitted


# ── Point assignment ───────────────────────────────────────────────────────

def assign_points_to_tracks(points, fitted_tracks):
    """Assign points to nearest vertical track within ±8in X influence.

    Displacement = X_eval - X_base (deviation from base X position).
    Rotation = dX/dZ from spline derivative.
    """
    N = len(points)
    px, pz = points[:, 0], points[:, 2]

    track_idx = np.full(N, -1, dtype=int)
    best_dist = np.full(N, np.inf)
    spline_dxdz = np.full(N, np.nan)
    displacement = np.full(N, np.nan)

    for i, track in enumerate(tqdm(fitted_tracks, desc="  Assigning points")):
        x_margin = INFLUENCE_HALF_X + np.ptp(track['x'])
        candidates = ((pz >= track['z_min']) & (pz <= track['z_max']) &
                       (px >= track['mean_x'] - x_margin) &
                       (px <= track['mean_x'] + x_margin))
        if not np.any(candidates):
            continue

        cand_idx = np.where(candidates)[0]
        z_range = track['z_max'] - track['z_min']

        if track['tck'] is not None and z_range > 0:
            tck = track['tck']
            u_pts = np.clip((pz[cand_idx] - track['z_min']) / z_range, 0, 1)
            x_eval, z_eval = splev(u_pts, tck)
            dx_du, dz_du = splev(u_pts, tck, der=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                dxdz = np.where(np.abs(dz_du) > 1e-10, dx_du / dz_du, 0.0)

            # Displacement relative to base X (bottom of spline)
            x_base, _ = splev(0.0, tck)
            disp = x_eval - x_base
        else:
            x_eval = np.full(len(cand_idx), track['mean_x'])
            dxdz = np.zeros(len(cand_idx))
            disp = np.zeros(len(cand_idx))

        x_dist = np.abs(px[cand_idx] - x_eval)
        in_zone = x_dist <= INFLUENCE_HALF_X
        update = in_zone & (x_dist < best_dist[cand_idx])

        upd_idx = cand_idx[update]
        best_dist[upd_idx] = x_dist[update]
        track_idx[upd_idx] = i
        spline_dxdz[upd_idx] = dxdz[update]
        displacement[upd_idx] = disp[update]

    return track_idx, spline_dxdz, displacement


# ── Plotting ───────────────────────────────────────────────────────────────

def _rasterize_values(points, values, track_idx, resolution, cmap_name, label,
                      title, output_path, symmetric=True):
    """Rasterize point values to a 2D image using bincount averaging."""
    assigned = track_idx >= 0
    if not np.any(assigned):
        print(f"  No points assigned, skipping {label} plot.")
        return

    px, pz = points[:, 0], points[:, 2]
    x_min, x_max = px.min(), px.max()
    z_min, z_max = pz.min(), pz.max()

    x_edges = np.arange(x_min, x_max + resolution, resolution)
    z_edges = np.arange(z_min, z_max + resolution, resolution)
    nc, nr = len(x_edges) - 1, len(z_edges) - 1

    a_px, a_pz = px[assigned], pz[assigned]
    a_vals = values[assigned]
    xi = np.clip(np.searchsorted(x_edges, a_px) - 1, 0, nc - 1)
    zi = np.clip(np.searchsorted(z_edges, a_pz) - 1, 0, nr - 1)
    flat = zi * nc + xi

    val_sum = np.bincount(flat, weights=a_vals, minlength=nr * nc)
    counts = np.bincount(flat, minlength=nr * nc)
    m = counts > 0
    raster = np.full(nr * nc, np.nan)
    raster[m] = val_sum[m] / counts[m]
    raster = raster.reshape(nr, nc)

    all_xi = np.clip(np.searchsorted(x_edges, px) - 1, 0, nc - 1)
    all_zi = np.clip(np.searchsorted(z_edges, pz) - 1, 0, nr - 1)
    all_flat = all_zi * nc + all_xi
    all_counts = np.bincount(all_flat, minlength=nr * nc).reshape(nr, nc)

    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    vals_flat = a_vals[~np.isnan(a_vals)]
    vmax = max(0.001, np.nanpercentile(np.abs(vals_flat), 95))

    fig, ax = plt.subplots(figsize=(16, 8))

    bg = np.where(all_counts > 0, 0.08, np.nan)
    ax.imshow(bg, aspect='auto', origin='lower', extent=extent,
              cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0)
    if symmetric:
        im = ax.imshow(raster, aspect='auto', origin='lower', extent=extent,
                       cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest')
    else:
        im = ax.imshow(raster, aspect='auto', origin='lower', extent=extent,
                       cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)

    ax.set_xlabel("X — along wall (m)")
    ax.set_ylabel("Z — elevation (m)")
    ax.set_title(title)
    ax.set_facecolor('black')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_joint_lines(raster, x_edges, z_edges, tracks, title_suffix, output_path):
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(raster, aspect='auto', origin='lower', extent=extent,
              cmap='hot', interpolation='nearest')

    for track in tracks:
        xs = [p[0] for p in track]
        zs = [p[1] for p in track]
        ax.plot(xs, zs, '-', color='cyan', linewidth=0.8, solid_capstyle='butt')

    ax.set_xlabel("X — along wall (m)")
    ax.set_ylabel("Z — elevation (m)")
    ax.set_title("Vertical Joints (%s) — %d tracks (min %d peaks, tol=%.3fm, search=%.2fm)" % (
        title_suffix, len(tracks), MIN_TRACK_PEAKS, MATCH_TOLERANCE, SEARCH_DISTANCE))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def run_vertical_pipeline(points, normals, raster, x_edges, z_edges, label, wall_id):
    """Run full vertical joint pipeline for one normal direction."""
    print(f"\n  --- {label} ---")

    print("  Detecting peaks...")
    detections = detect_peaks_vertical(raster, x_edges, z_edges)
    n_peaks = sum(len(d[1]) for d in detections)
    print(f"  {len(detections)} windows, {n_peaks} total peaks")

    print("  Linking peaks into tracks...")
    tracks = link_peaks_to_tracks(detections)
    print(f"  {len(tracks)} tracks (>= {MIN_TRACK_PEAKS} peaks)")

    tag = label.lower().replace('+', 'pos').replace('-', 'neg')
    plot_joint_lines(raster, x_edges, z_edges, tracks, label,
                     f"{OUTPUT_DIR}/wall_{wall_id}_vjoints_{tag}.png")

    print("  Fitting splines...")
    fitted = fit_tracks(tracks)
    n_splines = sum(1 for t in fitted if t['tck'] is not None)
    print(f"  {n_splines} tracks with splines")

    print("  Assigning points to tracks (±8in X influence)...")
    track_idx, dxdz, disp = assign_points_to_tracks(points, fitted)
    n_assigned = np.sum(track_idx >= 0)
    print(f"  {n_assigned}/{len(points)} points assigned")

    _rasterize_values(points, disp, track_idx, RASTER_RESOLUTION,
                      'coolwarm', 'X displacement from base (m)',
                      f'Vertical Joint Displacement ({label}) — X(z) - X(base)',
                      f"{OUTPUT_DIR}/wall_{wall_id}_vdisp_{tag}.png",
                      symmetric=True)

    _rasterize_values(points, dxdz, track_idx, RASTER_RESOLUTION,
                      'coolwarm', 'dX/dZ (rotation)',
                      f'Vertical Joint Rotation ({label}) — dX/dZ from spline',
                      f"{OUTPUT_DIR}/wall_{wall_id}_vrot_{tag}.png",
                      symmetric=True)


def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        print("No displacement PLY files found.")
        return

    ensure_dir(OUTPUT_DIR)

    for file in files:
        wall_id = os.path.basename(file).split("_")[1]
        print(f"\n=== Vertical Joint Debug: Wall {wall_id} ===")

        pc = o3d.t.io.read_point_cloud(file)
        points_full = pc.point.positions.numpy()
        print(f"  Full cloud: {len(points_full)} points")

        normals_full = get_normals(points_full, wall_id)

        # Crop to debug region
        center_x = DEBUG_CENTER_X
        if center_x is None:
            center_x = (points_full[:, 0].min() + points_full[:, 0].max()) / 2
        mask = ((points_full[:, 0] >= center_x - DEBUG_RANGE) &
                (points_full[:, 0] <= center_x + DEBUG_RANGE))
        points = points_full[mask]
        normals = normals_full[mask]
        print(f"  Debug region: X={center_x:.1f} +/- {DEBUG_RANGE:.1f}m -> {len(points)} points")

        if len(points) == 0:
            print("  No points, skipping.")
            continue

        print("  Rasterizing Nx+ and Nx-...")
        raster_pos, raster_neg, x_edges, z_edges = rasterize_vertical(
            points, normals, RASTER_RESOLUTION)
        print(f"  Raster: {raster_pos.shape}")

        run_vertical_pipeline(points, normals, raster_pos, x_edges, z_edges,
                              "Nx+", wall_id)
        run_vertical_pipeline(points, normals, raster_neg, x_edges, z_edges,
                              "Nx-", wall_id)

    print(f"\nDone. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
