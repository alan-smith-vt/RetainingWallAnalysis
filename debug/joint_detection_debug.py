"""
Debug visualization for joint detection.

Operates on a small region (CURVE_DEBUG_CENTER_X +/- CURVE_DEBUG_RANGE).
Produces a single image: Nz raster with detected peaks as dots,
each with a horizontal line showing the window it was detected in.

No block height constraints on peak detection — just raw find_peaks
with gaussian smoothing, min height, and a small min distance to
avoid double-counting the same joint.

Run from repo root: python debug/joint_detection_debug.py
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
PEAK_MIN_HEIGHT   = 0.03
PEAK_MIN_DIST_M   = 0.05   # just enough to avoid double-counting (5cm)
WINDOW_WIDTH      = 2.0
WINDOW_STEP       = 0.25

MATCH_TOLERANCE   = _cfg('JOINT_MATCH_TOLERANCE', 0.05)
MIN_TRACK_PEAKS   = 5

OUTPUT_DIR = "outputs/debug"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ── Pipeline ────────────────────────────────────────────────────────────────

NORMALS_CACHE_DIR = "outputs/point_clouds/unrolled"


def get_normals(points, wall_id, knn):
    """Load cached normals if available, otherwise compute and cache."""
    cache_path = os.path.join(NORMALS_CACHE_DIR, "normals_%s.ply" % wall_id)

    if os.path.exists(cache_path):
        print("  Loading cached normals from %s" % cache_path)
        pc = o3d.t.io.read_point_cloud(cache_path)
        cached_pts = pc.point.positions.numpy()
        if len(cached_pts) == len(points):
            return pc.point.normals.numpy()
        print("  Cache stale (%d vs %d points), recomputing" % (
            len(cached_pts), len(points)))

    print("  Computing normals (knn=%d)..." % knn)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    normals = np.asarray(pcd.normals)

    # Cache for reuse
    os.makedirs(NORMALS_CACHE_DIR, exist_ok=True)
    pcd_t = o3d.t.geometry.PointCloud()
    pcd_t.point.positions = points.astype(np.float32)
    pcd_t.point.normals = normals.astype(np.float32)
    o3d.t.io.write_point_cloud(cache_path, pcd_t)
    print("  Cached normals to %s" % cache_path)

    return normals


def rasterize(points, normals, resolution):
    x, z = points[:, 0], points[:, 2]
    x_edges = np.arange(x.min(), x.max() + resolution, resolution)
    z_edges = np.arange(z.min(), z.max() + resolution, resolution)

    xi = np.clip(np.searchsorted(x_edges, x) - 1, 0, len(x_edges) - 2)
    zi = np.clip(np.searchsorted(z_edges, z) - 1, 0, len(z_edges) - 2)

    nc, nr = len(x_edges) - 1, len(z_edges) - 1
    flat = zi * nc + xi
    total = nr * nc

    hz_sum = np.bincount(flat, weights=np.abs(normals[:, 2]), minlength=total)
    counts = np.bincount(flat, minlength=total)

    m = counts > 0
    raster_hz = np.zeros(total)
    raster_hz[m] = hz_sum[m] / counts[m]

    return raster_hz.reshape(nr, nc), x_edges, z_edges


def detect_peaks_windowed(raster_hz, x_edges, z_edges):
    """Detect peaks with no block height constraint.

    Returns list of (window_center_x, window_start_x, window_end_x, peak_z_array).
    """
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    res = RASTER_RESOLUTION

    min_dist_bins = max(1, int(PEAK_MIN_DIST_M / res))
    win_cols = max(1, int(WINDOW_WIDTH / res))
    step_cols = max(1, int(WINDOW_STEP / res))

    detections = []
    col = 0
    while col < raster_hz.shape[1]:
        ce = min(col + win_cols, raster_hz.shape[1])
        avg = np.mean(raster_hz[:, col:ce], axis=1)
        cx = np.mean(x_c[col:ce])
        wx_start = x_c[col]
        wx_end = x_c[min(ce - 1, len(x_c) - 1)]

        if len(avg) >= 3:
            sm = gaussian_filter1d(avg, sigma=GAUSSIAN_SIGMA)
            pi, _ = find_peaks(sm, height=PEAK_MIN_HEIGHT, distance=min_dist_bins)
            if len(pi) > 0:
                detections.append((cx, wx_start, wx_end, z_c[pi]))
        col += step_cols

    return detections


# ── Tracking ────────────────────────────────────────────────────────────────

def link_peaks_to_tracks(detections):
    """Link peaks between adjacent windows only.

    Each window's peaks are matched against the previous window's peaks.
    A track extends only if the very next window has a match within
    MATCH_TOLERANCE in Z. If a window has no match, the track ends.
    Unmatched peaks start new tracks.

    Returns list of tracks, each a list of (x, z) tuples.
    """
    if not detections:
        return []

    finished_tracks = []

    # Initialize with first window's peaks
    # Each active track is just a list of (x, z) points
    cx0, _, _, pz0 = detections[0]
    active = [[(cx0, z)] for z in pz0]

    for det_idx in range(1, len(detections)):
        cx, _, _, peak_zs = detections[det_idx]
        new_active = []
        used_p = set()

        if active and len(peak_zs) > 0:
            # Build distance matrix: active tracks vs current peaks
            active_z = np.array([t[-1][1] for t in active])
            dists = np.abs(active_z[:, None] - peak_zs[None, :])

            used_t = set()
            for fi in np.argsort(dists.ravel()):
                ti, pi = divmod(int(fi), len(peak_zs))
                if ti in used_t or pi in used_p:
                    continue
                if dists[ti, pi] > MATCH_TOLERANCE:
                    break
                # Extend this track
                active[ti].append((cx, peak_zs[pi]))
                new_active.append(active[ti])
                used_t.add(ti)
                used_p.add(pi)

            # Unmatched active tracks are finished
            for ti in range(len(active)):
                if ti not in used_t:
                    finished_tracks.append(active[ti])
        else:
            # No peaks or no active tracks — all active tracks end
            finished_tracks.extend(active)

        # Unmatched peaks start new tracks
        for pi in range(len(peak_zs)):
            if pi not in used_p:
                new_active.append([(cx, peak_zs[pi])])

        active = new_active

    # Flush remaining active tracks
    finished_tracks.extend(active)

    return [t for t in finished_tracks if len(t) >= MIN_TRACK_PEAKS]


# ── Spline fitting and field computation ────────────────────────────────────

SPLINE_SMOOTHING = _cfg('JOINT_SPLINE_SMOOTHING', 1.0)


def fit_tracks(tracks):
    """Fit a smoothed spline to each track.

    Returns list of dicts with:
        'x', 'z': raw arrays
        'x_min', 'x_max': horizontal extent
        'tck': spline coefficients (or None)
        'mean_z': mean elevation for sorting
    """
    fitted = []
    for track in tracks:
        x = np.array([p[0] for p in track])
        z = np.array([p[1] for p in track])
        entry = {
            'x': x, 'z': z,
            'x_min': x.min(), 'x_max': x.max(),
            'mean_z': z.mean(),
            'z_start': z[0],
            'tck': None,
        }
        if len(x) >= 4:
            try:
                tck, _ = splprep([x, z], s=SPLINE_SMOOTHING)
                entry['tck'] = tck
            except Exception:
                pass
        fitted.append(entry)

    # Sort by mean Z so we can find adjacent tracks
    fitted.sort(key=lambda t: t['mean_z'])
    return fitted


INFLUENCE_HALF_Z = 8 * FEET_TO_METERS / 12  # ±8 inches in meters


# Shared memory arrays set by _init_worker
_shared_px = None
_shared_pz = None


def _init_worker(px_shm_name, pz_shm_name, shape, dtype):
    """Attach to shared memory in each worker."""
    global _shared_px, _shared_pz
    from multiprocessing.shared_memory import SharedMemory
    px_shm = SharedMemory(name=px_shm_name)
    pz_shm = SharedMemory(name=pz_shm_name)
    _shared_px = np.ndarray(shape, dtype=dtype, buffer=px_shm.buf)
    _shared_pz = np.ndarray(shape, dtype=dtype, buffer=pz_shm.buf)


def _process_track(args):
    """Worker: evaluate one track against shared point arrays."""
    i, track = args
    px, pz = _shared_px, _shared_pz

    z_margin = INFLUENCE_HALF_Z + np.ptp(track['z'])
    candidates = ((px >= track['x_min']) & (px <= track['x_max']) &
                   (pz >= track['mean_z'] - z_margin) &
                   (pz <= track['mean_z'] + z_margin))
    if not np.any(candidates):
        return None

    cand_idx = np.where(candidates)[0]
    x_range = track['x_max'] - track['x_min']

    if track['tck'] is not None and x_range > 0:
        tck = track['tck']
        u_pts = np.clip((px[cand_idx] - track['x_min']) / x_range, 0, 1)
        x_eval, z_eval = splev(u_pts, tck)
        dx_du, dz_du = splev(u_pts, tck, der=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            dzdx = np.where(np.abs(dx_du) > 1e-10, dz_du / dx_du, 0.0)

        _, z_start = splev(0.0, tck)
        disp = z_eval - z_start
    else:
        z_eval = np.full(len(cand_idx), track['mean_z'])
        dzdx = np.zeros(len(cand_idx))
        disp = np.zeros(len(cand_idx))

    z_dist = np.abs(pz[cand_idx] - z_eval)
    in_zone = z_dist <= INFLUENCE_HALF_Z

    return (i, cand_idx[in_zone], z_dist[in_zone],
            dzdx[in_zone], disp[in_zone])


def assign_points_to_tracks(points, fitted_tracks):
    """Assign points to tracks, parallelized via shared memory.

    Returns:
        track_idx: (N,) int array, -1 if unassigned
        spline_dzdx: (N,) float array
        displacement: (N,) float array
    """
    from multiprocessing import Pool, cpu_count
    from multiprocessing.shared_memory import SharedMemory

    N = len(points)
    px = np.ascontiguousarray(points[:, 0], dtype=np.float64)
    pz = np.ascontiguousarray(points[:, 2], dtype=np.float64)

    # Create shared memory
    px_shm = SharedMemory(create=True, size=px.nbytes)
    pz_shm = SharedMemory(create=True, size=pz.nbytes)
    np.ndarray(px.shape, dtype=px.dtype, buffer=px_shm.buf)[:] = px
    np.ndarray(pz.shape, dtype=pz.dtype, buffer=pz_shm.buf)[:] = pz

    args = [(i, track) for i, track in enumerate(fitted_tracks)]
    n_workers = min(cpu_count(), len(fitted_tracks))
    print(f"  Using {n_workers} workers for {len(fitted_tracks)} tracks")

    results = []
    with Pool(n_workers, initializer=_init_worker,
              initargs=(px_shm.name, pz_shm.name, px.shape, px.dtype)) as pool:
        for r in tqdm(pool.imap_unordered(_process_track, args),
                      total=len(args), desc="  Assigning points"):
            if r is not None:
                results.append(r)

    # Clean up shared memory
    px_shm.close()
    px_shm.unlink()
    pz_shm.close()
    pz_shm.unlink()

    # Merge: closest track wins
    track_idx = np.full(N, -1, dtype=int)
    best_dist = np.full(N, np.inf)
    spline_dzdx = np.full(N, np.nan)
    displacement = np.full(N, np.nan)

    for (i, cand_idx, z_dist, dzdx, disp) in results:
        update = z_dist < best_dist[cand_idx]
        upd_idx = cand_idx[update]
        best_dist[upd_idx] = z_dist[update]
        track_idx[upd_idx] = i
        spline_dzdx[upd_idx] = dzdx[update]
        displacement[upd_idx] = disp[update]

    return track_idx, spline_dzdx, displacement


# ── Plot ────────────────────────────────────────────────────────────────────

def plot_joint_lines(raster_hz, x_edges, z_edges, tracks, output_path):
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
              cmap='hot', interpolation='nearest')

    for track in tracks:
        xs = [p[0] for p in track]
        zs = [p[1] for p in track]
        ax.plot(xs, zs, '-', color='white', linewidth=0.8, solid_capstyle='butt')

    ax.set_xlabel("X — along wall (m)")
    ax.set_ylabel("Z — elevation (m)")
    ax.set_title("Nz Raster + Joint Lines (min %d peaks, tol=%.3fm)\n"
                 "sigma=%.1f, min_height=%.3f, window=%.2fm, step=%.2fm" % (
                     MIN_TRACK_PEAKS, MATCH_TOLERANCE,
                     GAUSSIAN_SIGMA, PEAK_MIN_HEIGHT,
                     WINDOW_WIDTH, WINDOW_STEP))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _rasterize_values(points, values, track_idx, resolution, cmap_name, label,
                      title, output_path):
    """Rasterize point values to a 2D image using bincount averaging.

    Much faster than plt.scatter for millions of points.
    """
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

    # Raster of assigned values (average per cell)
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

    # Also build an occupancy raster (all points) for background
    all_xi = np.clip(np.searchsorted(x_edges, px) - 1, 0, nc - 1)
    all_zi = np.clip(np.searchsorted(z_edges, pz) - 1, 0, nr - 1)
    all_flat = all_zi * nc + all_xi
    all_counts = np.bincount(all_flat, minlength=nr * nc).reshape(nr, nc)

    # Render
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    vals_flat = a_vals[~np.isnan(a_vals)]
    vmax = max(0.001, np.nanpercentile(np.abs(vals_flat), 95))

    fig, ax = plt.subplots(figsize=(16, 8))

    # Dark background where points exist but no track assignment
    bg = np.where(all_counts > 0, 0.08, np.nan)
    ax.imshow(bg, aspect='auto', origin='lower', extent=extent,
              cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    # Colored overlay for assigned regions
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0)
    im = ax.imshow(raster, aspect='auto', origin='lower', extent=extent,
                   cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)

    ax.set_xlabel("X — along wall (m)")
    ax.set_ylabel("Z — elevation (m)")
    ax.set_title(title)
    ax.set_facecolor('black')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rotation(points, dzdx, track_idx, output_path):
    _rasterize_values(points, dzdx, track_idx, RASTER_RESOLUTION,
                      'coolwarm', 'dZ/dX (rotation)',
                      'Joint Rotation (dZ/dX from spline fit)', output_path)


def plot_displacement(points, disp, track_idx, output_path):
    _rasterize_values(points, disp, track_idx, RASTER_RESOLUTION,
                      'coolwarm', 'Displacement Z(x) - Z(start) (m)',
                      'Joint Displacement (Z relative to start of each joint)',
                      output_path)


def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        print("No displacement PLY files found.")
        return

    ensure_dir(OUTPUT_DIR)

    for file in files:
        wall_id = os.path.basename(file).split("_")[1]
        print(f"\n=== Joint Debug: Wall {wall_id} ===")

        pc = o3d.t.io.read_point_cloud(file)
        points_full = pc.point.positions.numpy()
        print(f"  Full cloud: {len(points_full)} points")

        # Get normals for full cloud (cached)
        normals_full = get_normals(points_full, wall_id, NORMAL_KNN)

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

        print("  Rasterizing...")
        raster_hz, x_edges, z_edges = rasterize(points, normals, RASTER_RESOLUTION)
        print(f"  Raster: {raster_hz.shape}")

        print("  Detecting peaks...")
        detections = detect_peaks_windowed(raster_hz, x_edges, z_edges)
        n_peaks = sum(len(d[3]) for d in detections)
        print(f"  {len(detections)} windows, {n_peaks} total peaks")

        print("  Linking peaks into tracks...")
        tracks = link_peaks_to_tracks(detections)
        print(f"  {len(tracks)} tracks (>= {MIN_TRACK_PEAKS} peaks)")

        plot_joint_lines(raster_hz, x_edges, z_edges, tracks,
                         f"{OUTPUT_DIR}/wall_{wall_id}_joints.png")

        print("  Fitting splines...")
        fitted = fit_tracks(tracks)
        print(f"  {sum(1 for t in fitted if t['tck'] is not None)} tracks with splines")

        print("  Assigning points to tracks (±8in influence)...")
        track_idx, dzdx, disp = assign_points_to_tracks(points, fitted)
        n_assigned = np.sum(track_idx >= 0)
        print(f"  {n_assigned}/{len(points)} points assigned")

        plot_rotation(points, dzdx, track_idx,
                      f"{OUTPUT_DIR}/wall_{wall_id}_rotation.png")
        plot_displacement(points, disp, track_idx,
                          f"{OUTPUT_DIR}/wall_{wall_id}_displacement.png")

    print(f"\nDone. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
