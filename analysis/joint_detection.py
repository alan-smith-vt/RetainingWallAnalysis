"""
Joint detection — full wall pipeline.

Computes normals (cached to PLY for reuse), rasterizes vertical normal
scores, detects and tracks horizontal joints, and outputs:
  - Point cloud with normals as scalar fields
  - Station-split images: Nz raster only, and Nz raster + joint lines

Run from repo root: python analysis/joint_detection.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from io import BytesIO
import cv2
import glob
from datetime import datetime

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
PEAK_MIN_HEIGHT     = 0.03           # per user request
WINDOW_WIDTH        = 2.0            # per user request
WINDOW_STEP         = 0.25           # per user request
MATCH_TOLERANCE     = _cfg('JOINT_MATCH_TOLERANCE', 0.05)
MIN_TRACK_LENGTH    = _cfg('JOINT_MIN_TRACK_LENGTH', 5)
SPLINE_SMOOTHING    = _cfg('JOINT_SPLINE_SMOOTHING', 1.0)
SPLINE_POINTS       = _cfg('JOINT_SPLINE_POINTS', 500)
BLOCK_HEIGHT_IN     = _cfg('BLOCK_HEIGHT_IN', 8)
BLOCK_HEIGHT_TOL    = _cfg('BLOCK_HEIGHT_TOLERANCE', 0.3)
BLOCK_HEIGHT_M      = BLOCK_HEIGHT_IN * FEET_TO_METERS / 12

WALL_IDS            = _cfg('WALL_IDS', [1])
STATION_MAX_FT      = _cfg('STATION_MAX_FT', None)
STATION_START_OFF   = _cfg('STATION_START_OFFSET_IN', 0)
STATION_END_OFF     = _cfg('STATION_END_OFFSET_IN', 0)
STATION_SPLITS      = _cfg('STATION_SPLITS', None)
RENDER_RESOLUTION   = _cfg('RENDER_RESOLUTION', 100)
RENDER_DPI          = _cfg('RENDER_DPI', 10)

NORMALS_CACHE_DIR   = "outputs/point_clouds/unrolled"
IMAGE_DIR           = "outputs/images"


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def ensure_dir(path):
    os.makedirs(os.path.dirname(path) if '.' in os.path.basename(path) else path,
                exist_ok=True)


# ── Normals (cached) ────────────────────────────────────────────────────────

def get_normals(points, wall_id):
    """Load cached normals or compute and save."""
    cache_path = os.path.join(NORMALS_CACHE_DIR, "normals_%s.ply" % wall_id)

    if os.path.exists(cache_path):
        printf("Loading cached normals from %s" % cache_path)
        pc = o3d.t.io.read_point_cloud(cache_path)
        cached_pts = pc.point.positions.numpy()
        # Verify same point count
        if len(cached_pts) == len(points):
            normals = pc.point.normals.numpy()
            printf("  Loaded %d normals" % len(normals))
            return normals
        else:
            printf("  Cache stale (%d vs %d points), recomputing" % (
                len(cached_pts), len(points)))

    printf("Computing normals (knn=%d, %d points)..." % (NORMAL_KNN, len(points)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    normals = np.asarray(pcd.normals)

    # Save with normals
    printf("  Caching normals to %s" % cache_path)
    ensure_dir(cache_path)
    pcd_t = o3d.t.geometry.PointCloud()
    pcd_t.point.positions = points.astype(np.float32)
    pcd_t.point.normals = normals.astype(np.float32)
    o3d.t.io.write_point_cloud(cache_path, pcd_t)

    return normals


# ── Write PLY with scalar fields ────────────────────────────────────────────

def write_ply_with_scalars(points, filepath, scalar_dict):
    """Write PLY with multiple float scalar fields."""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)

    header = "ply\nformat binary_little_endian 1.0\n"
    header += "element vertex %d\n" % n
    header += "property float x\nproperty float y\nproperty float z\n"

    scalar_arrays = []
    for name, arr in scalar_dict.items():
        header += "property float %s\n" % name
        scalar_arrays.append(np.asarray(arr, dtype=np.float32).ravel())

    header += "end_header\n"

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(points[i].tobytes())
            for sa in scalar_arrays:
                f.write(sa[i:i+1].tobytes())


# ── Pipeline functions ──────────────────────────────────────────────────────

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
    vt_sum = np.bincount(flat, weights=np.abs(normals[:, 0]), minlength=total)
    counts = np.bincount(flat, minlength=total)

    m = counts > 0
    raster_hz = np.zeros(total)
    raster_vt = np.zeros(total)
    raster_hz[m] = hz_sum[m] / counts[m]
    raster_vt[m] = vt_sum[m] / counts[m]

    return raster_hz.reshape(nr, nc), raster_vt.reshape(nr, nc), x_edges, z_edges


def detect_and_track(raster_hz, x_edges, z_edges):
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    res = RASTER_RESOLUTION

    esp_bins = max(1, int(BLOCK_HEIGHT_M / res))
    min_dist = max(1, int(esp_bins * (1 - BLOCK_HEIGHT_TOL)))
    win_cols = max(1, int(WINDOW_WIDTH / res))
    step_cols = max(1, int(WINDOW_STEP / res))

    # Detect
    detections = []
    col = 0
    while col < raster_hz.shape[1]:
        ce = min(col + win_cols, raster_hz.shape[1])
        avg = np.mean(raster_hz[:, col:ce], axis=1)
        cx = np.mean(x_c[col:ce])

        if len(avg) >= 3:
            sm = gaussian_filter1d(avg, sigma=GAUSSIAN_SIGMA)
            pi, _ = find_peaks(sm, height=PEAK_MIN_HEIGHT, distance=min_dist)
            if len(pi) > 0:
                detections.append((cx, z_c[pi], sm[pi]))
        col += step_cols

    # Track
    tracks = []
    active_z = []
    for x, pz, ps in detections:
        used_t, used_p = set(), set()
        if active_z:
            az = np.array(active_z)
            d = np.abs(az[:, None] - pz[None, :])
            for fi in np.argsort(d.ravel()):
                ti, pi = divmod(int(fi), len(pz))
                if ti in used_t or pi in used_p:
                    continue
                if d[ti, pi] > MATCH_TOLERANCE:
                    break
                tracks[ti]['x'].append(x)
                tracks[ti]['z'].append(pz[pi])
                active_z[ti] = pz[pi]
                used_t.add(ti)
                used_p.add(pi)
        for pi in range(len(pz)):
            if pi not in used_p:
                tracks.append({'x': [x], 'z': [pz[pi]]})
                active_z.append(pz[pi])

    # Filter
    tracks = [t for t in tracks if len(t['x']) >= MIN_TRACK_LENGTH]

    # Classify
    for t in tracks:
        t['mz'] = np.mean(t['z'])
    tracks.sort(key=lambda t: t['mz'])
    for i, t in enumerate(tracks):
        nb = False
        if i > 0:
            g = abs(tracks[i-1]['mz'] - t['mz'])
            n = round(g / BLOCK_HEIGHT_M) if BLOCK_HEIGHT_M > 0 else 0
            if n > 0 and abs(g - n * BLOCK_HEIGHT_M) <= BLOCK_HEIGHT_M * BLOCK_HEIGHT_TOL:
                nb = True
        if i < len(tracks) - 1:
            g = abs(t['mz'] - tracks[i+1]['mz'])
            n = round(g / BLOCK_HEIGHT_M) if BLOCK_HEIGHT_M > 0 else 0
            if n > 0 and abs(g - n * BLOCK_HEIGHT_M) <= BLOCK_HEIGHT_M * BLOCK_HEIGHT_TOL:
                nb = True
        t['label'] = 'joint' if nb else 'crack'

    # Fit splines (joints only)
    joint_tracks = [t for t in tracks if t['label'] == 'joint']
    splines = []
    for t in joint_tracks:
        x, z = np.array(t['x']), np.array(t['z'])
        if len(x) < 4:
            splines.append({'x': x, 'z': z, 'ok': False})
            continue
        try:
            tck, _ = splprep([x, z], s=SPLINE_SMOOTHING)
            u = np.linspace(0, 1, SPLINE_POINTS)
            xs, zs = splev(u, tck)
            splines.append({'x': xs, 'z': zs, 'ok': True})
        except Exception:
            splines.append({'x': x, 'z': z, 'ok': False})

    return tracks, splines


# ── Station alignment (mirrors rendering/point_cloud.py) ────────────────────

def station_align_raster(x_edges, points_x):
    """Map raster x_edges into station space. Returns (x_edges_aligned, total_m) or (x_edges, None)."""
    if STATION_MAX_FT is None:
        return x_edges, None
    total_m = STATION_MAX_FT * FEET_TO_METERS
    start_m = STATION_START_OFF * 0.0254
    end_m = STATION_END_OFF * 0.0254
    data_range_m = total_m - start_m - end_m

    x_min, x_max = points_x.min(), points_x.max()
    x_span = x_max - x_min
    if x_span <= 0:
        return x_edges, None

    scale = data_range_m / x_span
    aligned = (x_edges - x_min) * scale + start_m
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


# ── Image rendering ─────────────────────────────────────────────────────────

def render_raster_image(raster_hz, x_edges, z_edges, splines=None):
    """Render raster as image. If splines provided, overlay as 1px lines."""
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    x_range = x_c[-1] - x_c[0]
    z_range = z_c[-1] - z_c[0]
    fig_w = max(4, x_range * RENDER_RESOLUTION / RENDER_DPI)
    fig_h = max(2, z_range * RENDER_RESOLUTION / RENDER_DPI)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.dpi = RENDER_DPI

    ax.imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
              cmap='hot', interpolation='nearest')

    if splines is not None:
        for sp in splines:
            if sp['ok']:
                ax.plot(sp['x'], sp['z'], color='cyan', linewidth=0.5, solid_capstyle='butt')

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


def save_station_splits(img, x_edges, basename, suffix=""):
    """Split image by station ranges and save."""
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    x_min, x_max = x_c[0], x_c[-1]
    img_w = img.shape[1]

    ranges = get_station_ranges()

    if ranges:
        total_extent = x_c[-1]  # already in station space
        for start_m, end_m, label in ranges:
            # Map station range to pixel columns
            frac_start = (start_m - x_min) / (x_max - x_min)
            frac_end = (end_m - x_min) / (x_max - x_min)
            px_start = max(0, int(frac_start * img_w))
            px_end = min(img_w, int(frac_end * img_w))
            if px_end <= px_start:
                continue
            sub = img[:, px_start:px_end]
            path = "%s/%s%s_%s.png" % (IMAGE_DIR, basename, suffix, label)
            ensure_dir(path)
            cv2.imwrite(path, sub)
            printf("    Saved %s" % path)
    else:
        path = "%s/%s%s.png" % (IMAGE_DIR, basename, suffix)
        ensure_dir(path)
        cv2.imwrite(path, img)
        printf("    Saved %s" % path)


# ── Main ─────────────────────────────────────────────────────────────────────

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

        # Save point cloud with normal components as scalar fields
        scalar_path = os.path.join(NORMALS_CACHE_DIR, "normals_scalar_%s.ply" % wall_id)
        printf("Saving normals point cloud with scalar fields...")
        ensure_dir(scalar_path)
        write_ply_with_scalars(points, scalar_path, {
            'nx': normals[:, 0],
            'ny': normals[:, 1],
            'nz': normals[:, 2],
        })
        printf("  Saved %s" % scalar_path)

        # Rasterize
        printf("Rasterizing (resolution=%.3fm)..." % RASTER_RESOLUTION)
        raster_hz, raster_vt, x_edges, z_edges = rasterize(points, normals, RASTER_RESOLUTION)
        printf("  Raster: %d x %d" % (raster_hz.shape[1], raster_hz.shape[0]))

        # Detect and track
        printf("Detecting joints (window=%.2fm, step=%.2fm, min_height=%.3f)..." % (
            WINDOW_WIDTH, WINDOW_STEP, PEAK_MIN_HEIGHT))
        tracks, splines = detect_and_track(raster_hz, x_edges, z_edges)
        n_joints = sum(1 for t in tracks if t['label'] == 'joint')
        n_cracks = sum(1 for t in tracks if t['label'] == 'crack')
        printf("  %d joints, %d cracks, %d splines" % (n_joints, n_cracks, len(splines)))

        # Align raster to station space for image output
        x_edges_aligned, total_m = station_align_raster(x_edges, points[:, 0])

        # Also align spline X coords
        if total_m is not None:
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            x_span = x_max - x_min
            start_m = STATION_START_OFF * 0.0254
            data_range = total_m - start_m - STATION_END_OFF * 0.0254
            scale = data_range / x_span if x_span > 0 else 1
            splines_aligned = []
            for sp in splines:
                sp_a = dict(sp)
                sp_a['x'] = (np.array(sp['x']) - x_min) * scale + start_m
                splines_aligned.append(sp_a)
        else:
            splines_aligned = splines

        # Render images
        basename = "joints_%s" % wall_id

        printf("Rendering Nz raster images...")
        img_raster = render_raster_image(raster_hz, x_edges_aligned, z_edges)
        save_station_splits(img_raster, x_edges_aligned, basename, "_nz")

        printf("Rendering Nz + joint lines images...")
        img_joints = render_raster_image(raster_hz, x_edges_aligned, z_edges,
                                         splines=splines_aligned)
        save_station_splits(img_joints, x_edges_aligned, basename, "_nz_joints")

        # Save spline point clouds
        for i, sp in enumerate(splines):
            pts = np.zeros((len(sp['x']), 3), dtype=np.float32)
            pts[:, 0] = sp['x']
            pts[:, 2] = sp['z']
            path = os.path.join(NORMALS_CACHE_DIR, "joint_%s_%02d.ply" % (wall_id, i))
            ensure_dir(path)
            pcd = o3d.t.geometry.PointCloud()
            pcd.point.positions = pts
            colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(pts), 1))
            pcd.point.colors = colors
            o3d.t.io.write_point_cloud(path, pcd)

        printf("  Saved %d joint spline point clouds" % len(splines))

    printf("Done.")


if __name__ == '__main__':
    main()
