"""
Joint detection analysis for retaining walls.

Detects horizontal mortar joints (and cracks) in point cloud data using
surface normal analysis.  For each wall the script:
  1. Loads the unrolled displacement point cloud.
  2. Computes surface normals and rasterises a horizontal joint score (|Nz|).
  3. Detects peaks in vertical profiles, tracks them across windows, and
     classifies each track as *joint* or *crack*.
  4. Maps the per-cell raster score back to every point as a scalar field.
  5. Writes PLY files (original + unrolled) with the scalar field so the
     standard rendering pipeline can produce station-based image exports.

Run from repo root:  python analysis/joint_detection.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
import glob
from datetime import datetime
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from config import (
    FEET_TO_METERS,
)

# ── Settings (import from config, with defaults) ────────────────────────────

def _cfg(name, default):
    try:
        mod = sys.modules.get('config') or __import__('config')
        return getattr(mod, name, default)
    except Exception:
        return default

NORMAL_KNN            = _cfg('JOINT_NORMAL_KNN', 30)
RASTER_RESOLUTION     = _cfg('JOINT_RASTER_RESOLUTION', 0.01)
GAUSSIAN_SIGMA        = _cfg('JOINT_GAUSSIAN_SIGMA', 3)
PEAK_MIN_HEIGHT       = _cfg('JOINT_PEAK_MIN_HEIGHT', 0.15)
WINDOW_WIDTH          = _cfg('JOINT_WINDOW_WIDTH', 1.0)
WINDOW_STEP           = _cfg('JOINT_WINDOW_STEP', 0.5)
MATCH_TOLERANCE       = _cfg('JOINT_MATCH_TOLERANCE', 0.05)
MIN_TRACK_LENGTH      = _cfg('JOINT_MIN_TRACK_LENGTH', 5)
BLOCK_HEIGHT_IN       = _cfg('BLOCK_HEIGHT_IN', 8)
BLOCK_HEIGHT_TOLERANCE = _cfg('BLOCK_HEIGHT_TOLERANCE', 0.3)

BLOCK_HEIGHT_M = BLOCK_HEIGHT_IN * FEET_TO_METERS / 12


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


# ── Pipeline functions ──────────────────────────────────────────────────────

def compute_normals(points, knn):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals)


def rasterize_joint_score(points, normals, resolution, axis_h=0, axis_v=2):
    """Rasterise |Nz| into a 2-D grid and return the grid plus bin edges."""
    x = points[:, axis_h]
    z = points[:, axis_v]

    x_edges = np.arange(x.min(), x.max() + resolution, resolution)
    z_edges = np.arange(z.min(), z.max() + resolution, resolution)

    xi = np.clip(np.searchsorted(x_edges, x) - 1, 0, len(x_edges) - 2)
    zi = np.clip(np.searchsorted(z_edges, z) - 1, 0, len(z_edges) - 2)

    n_cols = len(x_edges) - 1
    n_rows = len(z_edges) - 1
    flat_idx = zi * n_cols + xi

    nz_abs = np.abs(normals[:, axis_v])

    hz_sum = np.bincount(flat_idx, weights=nz_abs, minlength=n_rows * n_cols)
    counts = np.bincount(flat_idx, minlength=n_rows * n_cols)

    mask = counts > 0
    raster_hz = np.zeros(n_rows * n_cols)
    raster_hz[mask] = hz_sum[mask] / counts[mask]

    return raster_hz.reshape(n_rows, n_cols), x_edges, z_edges, xi, zi


def map_raster_to_points(raster, xi, zi):
    """Look up every point's raster cell and return per-point scalar values."""
    return raster[zi, xi]


def detect_peaks_in_column(column, z_centers, sigma, min_height,
                           min_distance_bins, expected_spacing_bins=None,
                           spacing_tolerance=None):
    if len(column) < 3:
        return np.array([]), np.array([]), column

    smoothed = gaussian_filter1d(column, sigma=sigma)

    if expected_spacing_bins is not None and spacing_tolerance is not None:
        min_dist = max(min_distance_bins,
                       int(expected_spacing_bins * (1 - spacing_tolerance)))
    else:
        min_dist = min_distance_bins

    peak_indices, _ = find_peaks(smoothed, height=min_height, distance=min_dist)

    if len(peak_indices) == 0:
        return np.array([]), np.array([]), smoothed

    return z_centers[peak_indices], smoothed[peak_indices], smoothed


def detect_joints_windowed(raster_hz, x_edges, z_edges, resolution):
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    expected_spacing_bins = max(1, int(BLOCK_HEIGHT_M / resolution))
    min_distance_bins = max(1, int(BLOCK_HEIGHT_M * 0.5 / resolution))
    window_cols = max(1, int(WINDOW_WIDTH / resolution))
    step_cols = max(1, int(WINDOW_STEP / resolution))

    detections = []
    col = 0
    while col < raster_hz.shape[1]:
        col_end = min(col + window_cols, raster_hz.shape[1])
        window_avg = np.mean(raster_hz[:, col:col_end], axis=1)
        center_x = np.mean(x_centers[col:col_end])

        peak_z, peak_scores, _ = detect_peaks_in_column(
            window_avg, z_centers, GAUSSIAN_SIGMA, PEAK_MIN_HEIGHT,
            min_distance_bins,
            expected_spacing_bins=expected_spacing_bins,
            spacing_tolerance=BLOCK_HEIGHT_TOLERANCE,
        )

        if len(peak_z) > 0:
            detections.append((center_x, peak_z, peak_scores))

        col += step_cols

    return detections


def track_joints(detections):
    tracks = []
    active_track_z = []

    for x, peak_z, peak_scores in detections:
        used_tracks = set()
        used_peaks = set()

        if active_track_z:
            track_z_arr = np.array(active_track_z)
            dists = np.abs(track_z_arr[:, None] - peak_z[None, :])

            for flat_idx in np.argsort(dists.ravel()):
                ti, pi = divmod(int(flat_idx), len(peak_z))
                if ti in used_tracks or pi in used_peaks:
                    continue
                if dists[ti, pi] > MATCH_TOLERANCE:
                    break
                tracks[ti]['x'].append(x)
                tracks[ti]['z'].append(peak_z[pi])
                tracks[ti]['score'].append(peak_scores[pi])
                active_track_z[ti] = peak_z[pi]
                used_tracks.add(ti)
                used_peaks.add(pi)

        for pi in range(len(peak_z)):
            if pi not in used_peaks:
                tracks.append({'x': [x], 'z': [peak_z[pi]], 'score': [peak_scores[pi]]})
                active_track_z.append(peak_z[pi])

    return tracks


def filter_tracks(tracks):
    return [t for t in tracks if len(t['x']) >= MIN_TRACK_LENGTH]


def classify_tracks(tracks):
    if not tracks:
        return []

    for t in tracks:
        t['mean_z'] = np.mean(t['z'])
    tracks_sorted = sorted(tracks, key=lambda t: t['mean_z'])

    expected = BLOCK_HEIGHT_M
    tolerance = BLOCK_HEIGHT_TOLERANCE

    def spacing_ok(z1, z2):
        gap = abs(z2 - z1)
        if expected <= 0:
            return True
        n = round(gap / expected)
        return n > 0 and abs(gap - n * expected) <= expected * tolerance

    n = len(tracks_sorted)
    for i, t in enumerate(tracks_sorted):
        has_neighbor = False
        if i > 0 and spacing_ok(tracks_sorted[i - 1]['mean_z'], t['mean_z']):
            has_neighbor = True
        if i < n - 1 and spacing_ok(t['mean_z'], tracks_sorted[i + 1]['mean_z']):
            has_neighbor = True
        t['label'] = 'joint' if has_neighbor else 'crack'

    return tracks_sorted


# ── PLY writing (reuses wall_analysis pattern) ──────────────────────────────

def _write_ply_with_scalars(points, filepath, colors, scalars):
    """Write PLY with scalar field stored as intensity property."""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)

    header = "ply\nformat binary_little_endian 1.0\n"
    header += "element vertex %d\n" % n
    header += "property float x\nproperty float y\nproperty float z\n"

    has_colors = colors is not None
    if has_colors:
        colors = np.asarray(colors)
        if colors.dtype in (np.float32, np.float64):
            if colors.max() <= 1.0:
                colors = (colors * 255).clip(0, 255).astype(np.uint8)
            else:
                colors = colors.clip(0, 255).astype(np.uint8)
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"

    scalar_key = next(iter(scalars))
    scalar_array = np.asarray(scalars[scalar_key], dtype=np.float32).ravel()
    header += "property float intensity\n"

    header += "end_header\n"

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(points[i].tobytes())
            if has_colors:
                f.write(colors[i].tobytes())
            f.write(scalar_array[i:i+1].tobytes())


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        printf("No displacement PLY files found in outputs/point_clouds/unrolled/")
        return

    for file in sorted(files):
        wall_id = os.path.basename(file).split("_")[1]
        printf(f"=== Joint Detection: Wall {wall_id} ===")

        # Load point cloud
        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.positions.numpy()
        colors = pc.point.colors.numpy()
        if colors.dtype in (np.float32, np.float64) and colors.max() > 1.0:
            colors = (colors / 255.0).clip(0, 1)

        printf(f"  Loaded {len(points)} points, "
               f"X=[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]")

        # Zero the point cloud (match rendering pipeline)
        origin = np.min(points, axis=0)
        points_zeroed = points - origin

        # Compute normals
        printf("  Computing normals (KNN=%d)..." % NORMAL_KNN)
        normals = compute_normals(points_zeroed, NORMAL_KNN)

        # Rasterise joint score
        printf("  Rasterising joint score (resolution=%.3fm)..." % RASTER_RESOLUTION)
        raster_hz, x_edges, z_edges, xi, zi = rasterize_joint_score(
            points_zeroed, normals, RASTER_RESOLUTION)
        printf(f"  Raster shape: {raster_hz.shape}")

        # Map raster score back to per-point scalar
        point_scores = map_raster_to_points(raster_hz, xi, zi)
        printf(f"  Score range: [{point_scores.min():.4f}, {point_scores.max():.4f}]")

        # Detect and track joints (for classification info)
        printf("  Detecting peaks...")
        detections = detect_joints_windowed(raster_hz, x_edges, z_edges,
                                           RASTER_RESOLUTION)
        printf(f"  {len(detections)} windows, "
               f"{sum(len(d[1]) for d in detections)} total peaks")

        printf("  Tracking and classifying...")
        tracks = track_joints(detections)
        tracks = filter_tracks(tracks)
        tracks = classify_tracks(tracks)

        joint_tracks = [t for t in tracks if t.get('label') == 'joint']
        crack_tracks = [t for t in tracks if t.get('label') == 'crack']
        printf(f"  {len(joint_tracks)} joints, {len(crack_tracks)} cracks")

        # Write unrolled PLY with joint score scalar field
        out_unrolled = f"outputs/point_clouds/unrolled/joints_{wall_id}.ply"
        ensure_dir(out_unrolled)
        printf(f"  Writing {out_unrolled}")
        _write_ply_with_scalars(
            points_zeroed, out_unrolled, colors,
            scalars={'joint_score': point_scores},
        )

        # Write original (non-zeroed) PLY with joint score scalar field
        out_original = f"outputs/point_clouds/original/joints_{wall_id}.ply"
        ensure_dir(out_original)
        printf(f"  Writing {out_original}")
        _write_ply_with_scalars(
            points, out_original, colors,
            scalars={'joint_score': point_scores},
        )

    printf("Done.")


if __name__ == "__main__":
    main()
