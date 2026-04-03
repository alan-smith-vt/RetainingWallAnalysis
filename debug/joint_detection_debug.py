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
import glob

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


# ── Plot ────────────────────────────────────────────────────────────────────

def plot_peaks_on_raster(raster_hz, x_edges, z_edges, detections, output_path):
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
              cmap='hot', interpolation='nearest')

    # Draw each detection: dot at peak, horizontal line for window extent
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (cx, wx_s, wx_e, peak_zs) in enumerate(detections):
        c = colors[i % len(colors)]
        for pz in peak_zs:
            ax.plot(cx, pz, '.', color=c, markersize=4, zorder=5)
            ax.plot([wx_s, wx_e], [pz, pz], '-', color=c, linewidth=0.5,
                    alpha=0.6, zorder=4)

    ax.set_xlabel("X — along wall (m)")
    ax.set_ylabel("Z — elevation (m)")
    ax.set_title("Nz Raster + Detected Peaks (dots) with Window Extent (lines)\n"
                 "sigma=%.1f, min_height=%.3f, min_dist=%.3fm, "
                 "window=%.2fm, step=%.2fm" % (
                     GAUSSIAN_SIGMA, PEAK_MIN_HEIGHT, PEAK_MIN_DIST_M,
                     WINDOW_WIDTH, WINDOW_STEP))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path}")


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

        plot_peaks_on_raster(raster_hz, x_edges, z_edges, detections,
                             f"{OUTPUT_DIR}/wall_{wall_id}_peaks.png")

    print(f"\nDone. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
