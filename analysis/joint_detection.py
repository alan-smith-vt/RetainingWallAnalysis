"""
Joint detection — full wall pipeline.

Detects horizontal joint lines using surface normals, tracks them across
sliding windows, fits splines, and computes settlement and rotation fields.

Outputs station-split images for:
  - settlement: Z_max - Z(x) per joint (blue=0, red=max)
  - rotation: dZ/dX from spline derivative (symmetric coolwarm)
  - joint_lines: Nz raster with tracked joint overlays

Run from repo root: python analysis/joint_detection.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from io import BytesIO
import cv2
import glob
from datetime import datetime
from tqdm import tqdm

from config import FEET_TO_METERS

# ── Config with defaults ────────────────────────────────────────────────────

def _cfg(name, default):
    try:
        mod = sys.modules.get('config') or __import__('config')
        return getattr(mod, name, default)
    except Exception:
        return default

NORMAL_KNN          = _cfg('JOINT_NORMAL_KNN', 30)
RASTER_RESOLUTION   = _cfg('JOINT_RASTER_RESOLUTION', 0.01)
GAUSSIAN_SIGMA      = _cfg('JOINT_GAUSSIAN_SIGMA', 3.0)
PEAK_MIN_HEIGHT     = 0.03
PEAK_MIN_DIST_M     = 0.05          # avoid double-counting (5 cm)
WINDOW_WIDTH        = 2.0
WINDOW_STEP         = 0.25
MATCH_TOLERANCE     = _cfg('JOINT_MATCH_TOLERANCE', 0.05)
MIN_TRACK_PEAKS     = 5
SPLINE_SMOOTHING    = _cfg('JOINT_SPLINE_SMOOTHING', 1.0)

WALL_IDS            = _cfg('WALL_IDS', [1])
STATION_MAX_FT      = _cfg('STATION_MAX_FT', None)
STATION_START_OFF   = _cfg('STATION_START_OFFSET_IN', 0)
STATION_END_OFF     = _cfg('STATION_END_OFFSET_IN', 0)
STATION_SPLITS      = _cfg('STATION_SPLITS', None)
RENDER_RESOLUTION   = _cfg('RENDER_RESOLUTION', 100)
RENDER_DPI          = _cfg('RENDER_DPI', 10)
MARKER_SIZE         = _cfg('MARKER_SIZE_DISPLACEMENTS', 500)

NORMALS_CACHE_DIR   = "outputs/point_clouds/unrolled"
IMAGE_DIR           = "outputs/images"

INFLUENCE_HALF_Z    = 8 * FEET_TO_METERS / 12  # ±8 inches in meters


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath) if '.' in os.path.basename(filepath) else filepath,
                exist_ok=True)


# ── Normals (cached) ────────────────────────────────────────────────────────

def get_normals(points, wall_id):
    """Load cached normals or compute and save."""
    cache_path = os.path.join(NORMALS_CACHE_DIR, "normals_%s.ply" % wall_id)

    if os.path.exists(cache_path):
        printf("Loading cached normals from %s" % cache_path)
        pc = o3d.t.io.read_point_cloud(cache_path)
        cached_pts = pc.point.positions.numpy()
        if len(cached_pts) == len(points):
            return pc.point.normals.numpy()
        printf("  Cache stale (%d vs %d points), recomputing" % (
            len(cached_pts), len(points)))

    printf("Computing normals (knn=%d, %d points)..." % (NORMAL_KNN, len(points)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    normals = np.asarray(pcd.normals)

    printf("  Caching normals to %s" % cache_path)
    ensure_dir(cache_path)
    pcd_t = o3d.t.geometry.PointCloud()
    pcd_t.point.positions = points.astype(np.float32)
    pcd_t.point.normals = normals.astype(np.float32)
    o3d.t.io.write_point_cloud(cache_path, pcd_t)

    return normals


# ── Pipeline ────────────────────────────────────────────────────────────────

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
    """Detect peaks — no block height constraint, just raw find_peaks."""
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

        if len(avg) >= 3:
            sm = gaussian_filter1d(avg, sigma=GAUSSIAN_SIGMA)
            pi, _ = find_peaks(sm, height=PEAK_MIN_HEIGHT, distance=min_dist_bins)
            if len(pi) > 0:
                detections.append((cx, z_c[pi]))
        col += step_cols

    return detections


def link_peaks_to_tracks(detections):
    """Link peaks between adjacent windows only.

    A track extends only if the very next window has a match within
    MATCH_TOLERANCE in Z. If a window has no match, the track ends.
    """
    if not detections:
        return []

    finished_tracks = []
    cx0, pz0 = detections[0]
    active = [[(cx0, z)] for z in pz0]

    for det_idx in range(1, len(detections)):
        cx, peak_zs = detections[det_idx]
        new_active = []
        used_p = set()

        if active and len(peak_zs) > 0:
            active_z = np.array([t[-1][1] for t in active])
            dists = np.abs(active_z[:, None] - peak_zs[None, :])

            used_t = set()
            for fi in np.argsort(dists.ravel()):
                ti, pi = divmod(int(fi), len(peak_zs))
                if ti in used_t or pi in used_p:
                    continue
                if dists[ti, pi] > MATCH_TOLERANCE:
                    break
                active[ti].append((cx, peak_zs[pi]))
                new_active.append(active[ti])
                used_t.add(ti)
                used_p.add(pi)

            for ti in range(len(active)):
                if ti not in used_t:
                    finished_tracks.append(active[ti])
        else:
            finished_tracks.extend(active)

        for pi in range(len(peak_zs)):
            if pi not in used_p:
                new_active.append([(cx, peak_zs[pi])])

        active = new_active

    finished_tracks.extend(active)
    return [t for t in finished_tracks if len(t) >= MIN_TRACK_PEAKS]


def fit_tracks(tracks):
    """Fit a smoothed spline to each track."""
    fitted = []
    for track in tracks:
        x = np.array([p[0] for p in track])
        z = np.array([p[1] for p in track])
        entry = {
            'x': x, 'z': z,
            'x_min': x.min(), 'x_max': x.max(),
            'mean_z': z.mean(),
            'tck': None,
        }
        if len(x) >= 4:
            try:
                tck, _ = splprep([x, z], s=SPLINE_SMOOTHING)
                entry['tck'] = tck
            except Exception:
                pass
        fitted.append(entry)

    fitted.sort(key=lambda t: t['mean_z'])
    return fitted


def assign_points_to_tracks(points, fitted_tracks):
    """Assign each point to its nearest track within ±8in influence zone.

    Settlement = Z_max - Z(x) per track (relative to highest point, always >= 0).
    Rotation = dZ/dX from spline derivative.
    """
    N = len(points)
    px, pz = points[:, 0], points[:, 2]

    track_idx = np.full(N, -1, dtype=int)
    best_dist = np.full(N, np.inf)
    spline_dzdx = np.full(N, np.nan)
    settlement = np.full(N, np.nan)

    for i, track in enumerate(tqdm(fitted_tracks, desc="  Assigning points")):
        z_margin = INFLUENCE_HALF_Z + np.ptp(track['z'])
        candidates = ((px >= track['x_min']) & (px <= track['x_max']) &
                       (pz >= track['mean_z'] - z_margin) &
                       (pz <= track['mean_z'] + z_margin))
        if not np.any(candidates):
            continue

        cand_idx = np.where(candidates)[0]
        x_range = track['x_max'] - track['x_min']

        if track['tck'] is not None and x_range > 0:
            tck = track['tck']
            u_pts = np.clip((px[cand_idx] - track['x_min']) / x_range, 0, 1)
            x_eval, z_eval = splev(u_pts, tck)
            dx_du, dz_du = splev(u_pts, tck, der=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                dzdx = np.where(np.abs(dx_du) > 1e-10, dz_du / dx_du, 0.0)

            # Settlement relative to highest point on this spline
            u_dense = np.linspace(0, 1, 200)
            _, z_dense = splev(u_dense, tck)
            z_max = z_dense.max()
            disp = z_max - z_eval  # always >= 0
        else:
            z_eval = np.full(len(cand_idx), track['mean_z'])
            dzdx = np.zeros(len(cand_idx))
            disp = np.zeros(len(cand_idx))

        z_dist = np.abs(pz[cand_idx] - z_eval)
        in_zone = z_dist <= INFLUENCE_HALF_Z
        update = in_zone & (z_dist < best_dist[cand_idx])

        upd_idx = cand_idx[update]
        best_dist[upd_idx] = z_dist[update]
        track_idx[upd_idx] = i
        spline_dzdx[upd_idx] = dzdx[update]
        settlement[upd_idx] = disp[update]

    return track_idx, spline_dzdx, settlement


# ── Station alignment ──────────────────────────────────────────────────────

def station_align(points_x):
    """Map x-coordinates to station space. Returns (x_aligned, total_m) or (points_x, None)."""
    if STATION_MAX_FT is None:
        return points_x, None
    total_m = STATION_MAX_FT * FEET_TO_METERS
    start_m = STATION_START_OFF * 0.0254
    end_m = STATION_END_OFF * 0.0254
    data_range_m = total_m - start_m - end_m

    x_min, x_max = points_x.min(), points_x.max()
    x_span = x_max - x_min
    if x_span <= 0:
        return points_x, None

    scale = data_range_m / x_span
    aligned = (points_x - x_min) * scale + start_m
    return aligned, total_m


def get_station_ranges():
    if not STATION_SPLITS:
        return None
    ranges = []
    for start_ft, end_ft in STATION_SPLITS:
        start_m = start_ft * FEET_TO_METERS
        end_m = end_ft * FEET_TO_METERS
        label = "%d+%02d_to_%d+%02d" % (
            int(start_ft) // 100, int(start_ft) % 100,
            int(end_ft) // 100, int(end_ft) % 100)
        ranges.append((start_m, end_m, label))
    return ranges


# ── Image rendering (matches rendering/point_cloud.py approach) ────────────

def render_scatter(points_x, points_z, colors, x_extent, z_extent, sz):
    """Render colored scatter to a BGRA image, matching point_cloud.py style."""
    fig = plt.figure()
    ax = plt.gca()
    fig.dpi = RENDER_DPI
    resolution = RENDER_RESOLUTION

    s = [x_extent * 100, z_extent * 100]
    fig.set_size_inches((s[0] * resolution / 100 + 10) / fig.dpi,
                        (s[1] * resolution / 100 + 10) / fig.dpi)
    ax.set_xlim((0, s[0]))
    ax.set_ylim((0, s[1]))
    ax.scatter(points_x * 100, points_z * 100, c=colors, marker=',', lw=0,
               s=(sz / fig.dpi) ** 2)
    ax.axis('off')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, dpi=fig.dpi, transparent=True, format='png')
    buf.seek(0)

    data = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    res = img[5:-5, 5:-5]

    plt.close()
    plt.clf()
    plt.close(fig)
    return res


def render_joint_lines(raster_hz, x_edges, z_edges, tracks, x_extent, z_extent):
    """Render Nz raster + joint line overlays."""
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    s = [x_extent * 100, z_extent * 100]
    fig = plt.figure()
    ax = plt.gca()
    fig.dpi = RENDER_DPI
    resolution = RENDER_RESOLUTION
    fig.set_size_inches((s[0] * resolution / 100 + 10) / fig.dpi,
                        (s[1] * resolution / 100 + 10) / fig.dpi)

    ax.imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
              cmap='hot', interpolation='nearest')

    for track in tracks:
        xs = [p[0] for p in track]
        zs = [p[1] for p in track]
        ax.plot(xs, zs, '-', color='cyan', linewidth=0.5, solid_capstyle='butt')

    ax.axis('off')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, dpi=fig.dpi, transparent=True, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    data = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return img


def save_station_images(points_x_aligned, points_z, colors, total_m, z_extent,
                        basename, sz=MARKER_SIZE):
    """Render and save station-split images (or full image if no splits)."""
    station_ranges = get_station_ranges()

    if station_ranges:
        for start_m, end_m, label in station_ranges:
            mask = (points_x_aligned >= start_m) & (points_x_aligned < end_m)
            sub_x = points_x_aligned[mask] - start_m
            sub_z = points_z[mask]
            sub_c = colors[mask]
            if len(sub_x) == 0:
                continue
            x_ext = end_m - start_m
            img = render_scatter(sub_x, sub_z, sub_c, x_ext, z_extent, sz)
            path = "%s/%s_%s.png" % (IMAGE_DIR, basename, label)
            ensure_dir(path)
            cv2.imwrite(path, img)
            printf("    Saved %s" % path)
    else:
        x_ext = total_m if total_m else (points_x_aligned.max() - points_x_aligned.min())
        img = render_scatter(points_x_aligned, points_z, colors, x_ext, z_extent, sz)
        path = "%s/%s.png" % (IMAGE_DIR, basename)
        ensure_dir(path)
        cv2.imwrite(path, img)
        printf("    Saved %s" % path)


def save_joint_line_images(raster_hz, x_edges_aligned, z_edges, tracks_aligned,
                           total_m, z_extent, basename):
    """Render and save station-split joint line images."""
    x_c = (x_edges_aligned[:-1] + x_edges_aligned[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2

    station_ranges = get_station_ranges()

    if station_ranges:
        for start_m, end_m, label in station_ranges:
            # Crop raster columns to this station range
            col_mask = (x_c >= start_m) & (x_c < end_m)
            if not np.any(col_mask):
                continue
            col_idx = np.where(col_mask)[0]
            sub_raster = raster_hz[:, col_idx]
            sub_x_edges = np.append(x_edges_aligned[col_idx] - start_m,
                                    x_edges_aligned[col_idx[-1] + 1] - start_m)
            # Filter and shift tracks
            sub_tracks = []
            for track in tracks_aligned:
                pts = [(x - start_m, z) for x, z in track
                       if start_m <= x < end_m]
                if len(pts) >= 2:
                    sub_tracks.append(pts)

            x_ext = end_m - start_m
            img = render_joint_lines(sub_raster, sub_x_edges, z_edges,
                                     sub_tracks, x_ext, z_extent)
            path = "%s/%s_%s.png" % (IMAGE_DIR, basename, label)
            ensure_dir(path)
            cv2.imwrite(path, img)
            printf("    Saved %s" % path)
    else:
        x_ext = total_m if total_m else (x_c[-1] - x_c[0])
        img = render_joint_lines(raster_hz, x_edges_aligned, z_edges,
                                 tracks_aligned, x_ext, z_extent)
        path = "%s/%s.png" % (IMAGE_DIR, basename)
        ensure_dir(path)
        cv2.imwrite(path, img)
        printf("    Saved %s" % path)


# ── Colormaps ──────────────────────────────────────────────────────────────

def settlement_colors(values):
    """Blue (0) to Red (max settlement). All values >= 0."""
    cmap = plt.cm.coolwarm
    valid = ~np.isnan(values)
    vmax = max(0.001, np.nanpercentile(values[valid], 95)) if np.any(valid) else 0.001
    norm = Normalize(vmin=0, vmax=vmax)
    return cmap(norm(np.clip(values, 0, vmax)))[:, :3]


def rotation_colors(values):
    """Symmetric coolwarm around 0."""
    cmap = plt.cm.coolwarm
    valid = ~np.isnan(values)
    vmax = max(0.001, np.nanpercentile(np.abs(values[valid]), 95)) if np.any(valid) else 0.001
    norm = Normalize(vmin=-vmax, vmax=vmax)
    return cmap(norm(np.clip(values, -vmax, vmax)))[:, :3]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        printf("No displacement PLY files found.")
        sys.exit(1)

    for file in files:
        wall_id = os.path.basename(file).split("_")[1]
        printf("=== Wall %s: %s ===" % (wall_id, file))

        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.positions.numpy()
        printf("  %d points" % len(points))

        # Normals (cached)
        normals = get_normals(points, wall_id)

        # Rasterize
        printf("Rasterizing (resolution=%.3fm)..." % RASTER_RESOLUTION)
        raster_hz, x_edges, z_edges = rasterize(points, normals, RASTER_RESOLUTION)
        printf("  Raster: %d x %d" % (raster_hz.shape[1], raster_hz.shape[0]))

        # Detect peaks
        printf("Detecting peaks (window=%.2fm, step=%.2fm)..." % (WINDOW_WIDTH, WINDOW_STEP))
        detections = detect_peaks_windowed(raster_hz, x_edges, z_edges)
        n_peaks = sum(len(d[1]) for d in detections)
        printf("  %d windows, %d total peaks" % (len(detections), n_peaks))

        # Track
        printf("Linking peaks into tracks...")
        tracks = link_peaks_to_tracks(detections)
        printf("  %d tracks (>= %d peaks)" % (len(tracks), MIN_TRACK_PEAKS))

        # Fit splines
        printf("Fitting splines...")
        fitted = fit_tracks(tracks)
        n_splines = sum(1 for t in fitted if t['tck'] is not None)
        printf("  %d tracks with splines" % n_splines)

        # Assign points to tracks
        printf("Assigning points to tracks (±8in influence)...")
        track_idx, dzdx, settle = assign_points_to_tracks(points, fitted)
        n_assigned = np.sum(track_idx >= 0)
        printf("  %d/%d points assigned" % (n_assigned, len(points)))

        # ── Station alignment ──────────────────────────────────────────────
        px = points[:, 0].copy()
        pz = points[:, 2].copy()

        # Zero-base the Z coordinates for rendering
        pz_zeroed = pz - pz.min()
        z_extent = pz_zeroed.max()

        # Align X to station space
        px_aligned, total_m = station_align(px)

        # Also align raster edges and tracks for joint line images
        if total_m is not None:
            x_min, x_max = px.min(), px.max()
            x_span = x_max - x_min
            start_m = STATION_START_OFF * 0.0254
            data_range = total_m - start_m - STATION_END_OFF * 0.0254
            scale = data_range / x_span if x_span > 0 else 1
            x_edges_aligned = (x_edges - x_min) * scale + start_m
            tracks_aligned = []
            for track in tracks:
                tracks_aligned.append(
                    [((x - x_min) * scale + start_m, z) for x, z in track])
        else:
            x_edges_aligned = x_edges
            tracks_aligned = tracks

        # ── Render settlement images ───────────────────────────────────────
        assigned = track_idx >= 0
        printf("Rendering settlement images...")
        if np.any(assigned):
            colors_settle = np.zeros((len(points), 3))
            colors_settle[assigned] = settlement_colors(settle[assigned])
            # Unassigned points get transparent (won't show on transparent bg)
            # But scatter plots all points — use dark gray for unassigned
            colors_settle[~assigned] = [0.05, 0.05, 0.05]
            save_station_images(px_aligned, pz_zeroed, colors_settle, total_m,
                                z_extent, "settlement_%s" % wall_id)

        # ── Render rotation images ─────────────────────────────────────────
        printf("Rendering rotation images...")
        if np.any(assigned):
            colors_rot = np.zeros((len(points), 3))
            colors_rot[assigned] = rotation_colors(dzdx[assigned])
            colors_rot[~assigned] = [0.05, 0.05, 0.05]
            save_station_images(px_aligned, pz_zeroed, colors_rot, total_m,
                                z_extent, "rotation_%s" % wall_id)

        # ── Render joint line images ───────────────────────────────────────
        printf("Rendering joint line images...")
        z_edges_zeroed = z_edges - pz.min()
        save_joint_line_images(raster_hz, x_edges_aligned, z_edges_zeroed,
                               tracks_aligned, total_m, z_extent,
                               "joint_lines_%s" % wall_id)

        # ── Generate legends with actual data ranges ──────────────────────
        if np.any(assigned):
            from rendering.colorbar import create_settlement_colorbar, create_rotation_colorbar

            valid_settle = settle[assigned]
            valid_settle = valid_settle[~np.isnan(valid_settle)]
            settle_max_m = max(0.001, np.percentile(valid_settle, 95))

            valid_rot = dzdx[assigned]
            valid_rot = valid_rot[~np.isnan(valid_rot)]
            rot_max = max(0.001, np.percentile(np.abs(valid_rot), 95))

            create_settlement_colorbar(
                settle_max_m,
                save_path=os.path.join(IMAGE_DIR, "legend_settlement.png"))
            create_rotation_colorbar(
                rot_max,
                save_path=os.path.join(IMAGE_DIR, "legend_rotation.png"))

    printf("Done.")


if __name__ == '__main__':
    main()
