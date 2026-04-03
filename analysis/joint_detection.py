"""
Joint detection from surface normals.

Computes normals on the unrolled wall point cloud, rasterizes a 2D
"joint score" image, detects horizontal joint lines via peak finding
in vertical profiles, tracks joints across sliding windows, and fits
per-joint splines.

Designed for the downsampled point cloud first; full-resolution pipeline
will use the LOD Octree module later.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import glob
from datetime import datetime

from config import (
    FEET_TO_METERS,
    JOINT_NORMAL_KNN,
    JOINT_RASTER_RESOLUTION,
    JOINT_GAUSSIAN_SIGMA,
    JOINT_PEAK_MIN_HEIGHT,
    JOINT_WINDOW_WIDTH,
    JOINT_WINDOW_STEP,
    JOINT_MATCH_TOLERANCE,
    JOINT_MIN_TRACK_LENGTH,
    JOINT_SPLINE_SMOOTHING,
    JOINT_SPLINE_POINTS,
    BLOCK_HEIGHT_IN,
    BLOCK_WIDTH_IN,
    BLOCK_HEIGHT_TOLERANCE,
)

# Derived block dimensions in meters
BLOCK_HEIGHT_M = BLOCK_HEIGHT_IN * FEET_TO_METERS / 12  # inches → meters
BLOCK_WIDTH_M = BLOCK_WIDTH_IN * FEET_TO_METERS / 12


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ── Stage 1: Normals → 2D raster ────────────────────────────────────────────

def compute_normals(points, knn):
    """Estimate surface normals using Open3D KNN search.

    Args:
        points: (N, 3) array
        knn: number of neighbors for normal estimation

    Returns:
        normals: (N, 3) array of unit normals
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals)


def rasterize_joint_score(points, normals, resolution, axis_h=0, axis_v=2):
    """Project normal-based joint scores onto a regular 2D grid.

    Args:
        points: (N, 3) point positions
        normals: (N, 3) surface normals
        resolution: grid cell size in meters
        axis_h: horizontal axis index (0 = X, along wall)
        axis_v: vertical axis index (2 = Z, elevation)

    Returns:
        raster_hz: 2D array of horizontal joint score (|Nz|)
        raster_vt: 2D array of vertical joint score (|Nx|)
        x_edges: 1D array of horizontal bin edges
        z_edges: 1D array of vertical bin edges
    """
    x = points[:, axis_h]
    z = points[:, axis_v]

    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    x_edges = np.arange(x_min, x_max + resolution, resolution)
    z_edges = np.arange(z_min, z_max + resolution, resolution)

    # Bin indices for each point
    xi = np.clip(np.searchsorted(x_edges, x) - 1, 0, len(x_edges) - 2)
    zi = np.clip(np.searchsorted(z_edges, z) - 1, 0, len(z_edges) - 2)

    nx_abs = np.abs(normals[:, axis_h])
    nz_abs = np.abs(normals[:, axis_v])

    n_cols = len(x_edges) - 1
    n_rows = len(z_edges) - 1

    # Accumulate scores and counts using linear indexing
    flat_idx = zi * n_cols + xi
    raster_hz_sum = np.bincount(flat_idx, weights=nz_abs, minlength=n_rows * n_cols)
    raster_vt_sum = np.bincount(flat_idx, weights=nx_abs, minlength=n_rows * n_cols)
    counts = np.bincount(flat_idx, minlength=n_rows * n_cols)

    # Average per cell, avoid division by zero
    mask = counts > 0
    raster_hz = np.zeros(n_rows * n_cols)
    raster_vt = np.zeros(n_rows * n_cols)
    raster_hz[mask] = raster_hz_sum[mask] / counts[mask]
    raster_vt[mask] = raster_vt_sum[mask] / counts[mask]

    raster_hz = raster_hz.reshape(n_rows, n_cols)
    raster_vt = raster_vt.reshape(n_rows, n_cols)

    return raster_hz, raster_vt, x_edges, z_edges


# ── Stage 2: Peak detection in vertical profiles ────────────────────────────

def detect_peaks_in_column(column, z_centers, sigma, min_height,
                           min_distance_bins, expected_spacing_bins=None,
                           spacing_tolerance=None):
    """Find peaks in a single vertical profile of joint scores.

    Uses block height as a soft constraint: peaks closer than
    (1 - tolerance) * expected_spacing are suppressed.

    Args:
        column: 1D array of joint scores (one raster column)
        z_centers: 1D array of Z positions for each row
        sigma: gaussian smoothing sigma in bins
        min_height: minimum peak height after smoothing
        min_distance_bins: minimum distance between peaks in bins
        expected_spacing_bins: expected peak spacing from block height (optional)
        spacing_tolerance: fractional tolerance on spacing (optional)

    Returns:
        peak_z: array of Z positions of detected peaks
        peak_scores: array of scores at peaks
        smoothed: the smoothed column (for debug)
    """
    if len(column) < 3:
        return np.array([]), np.array([]), column

    smoothed = gaussian_filter1d(column, sigma=sigma)

    # Use block height to set minimum distance between peaks
    if expected_spacing_bins is not None and spacing_tolerance is not None:
        min_dist = max(min_distance_bins,
                       int(expected_spacing_bins * (1 - spacing_tolerance)))
    else:
        min_dist = min_distance_bins

    peak_indices, properties = find_peaks(
        smoothed,
        height=min_height,
        distance=min_dist,
    )

    if len(peak_indices) == 0:
        return np.array([]), np.array([]), smoothed

    return z_centers[peak_indices], smoothed[peak_indices], smoothed


# ── Stage 3: Joint tracking across windows ───────────────────────────────────

def detect_joints_windowed(raster_hz, x_edges, z_edges, window_width,
                           window_step, sigma, min_height, resolution):
    """Sweep along X, detect Z-peaks in each window's averaged column.

    Uses BLOCK_HEIGHT_M as a soft constraint on peak spacing.

    Args:
        raster_hz: 2D horizontal joint score raster (rows=Z, cols=X)
        x_edges, z_edges: bin edges from rasterization
        window_width: width of averaging window in meters
        window_step: step between windows in meters
        sigma: gaussian smoothing sigma in bins
        min_height: minimum peak height
        resolution: raster cell size in meters

    Returns:
        detections: list of (window_center_x, peak_z_array, peak_score_array)
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    expected_spacing_bins = max(1, int(BLOCK_HEIGHT_M / resolution))
    min_distance_bins = max(1, int(BLOCK_HEIGHT_M * 0.5 / resolution))

    window_cols = max(1, int(window_width / resolution))
    step_cols = max(1, int(window_step / resolution))

    detections = []
    col = 0
    while col < raster_hz.shape[1]:
        col_end = min(col + window_cols, raster_hz.shape[1])
        window_avg = np.mean(raster_hz[:, col:col_end], axis=1)
        center_x = np.mean(x_centers[col:col_end])

        peak_z, peak_scores, _ = detect_peaks_in_column(
            window_avg, z_centers, sigma, min_height, min_distance_bins,
            expected_spacing_bins=expected_spacing_bins,
            spacing_tolerance=BLOCK_HEIGHT_TOLERANCE,
        )

        if len(peak_z) > 0:
            detections.append((center_x, peak_z, peak_scores))

        col += step_cols

    return detections


def track_joints(detections, match_tolerance):
    """Link detected peaks across windows into continuous joint tracks.

    Uses greedy nearest-neighbor matching: for each detection window,
    match each peak to the nearest existing track (within tolerance).
    Unmatched peaks start new tracks.

    Args:
        detections: list of (x, peak_z_array, peak_score_array)
        match_tolerance: max Z distance to match a peak to an existing track

    Returns:
        tracks: list of dicts with keys:
            'x': list of X positions
            'z': list of Z positions
            'score': list of scores
    """
    tracks = []
    active_track_z = []  # current Z estimate for each track

    for x, peak_z, peak_scores in detections:
        used_tracks = set()
        used_peaks = set()

        if active_track_z:
            # Distance matrix: (n_tracks, n_peaks)
            track_z_arr = np.array(active_track_z)
            dists = np.abs(track_z_arr[:, None] - peak_z[None, :])

            # Greedy match — closest pairs first
            flat_order = np.argsort(dists.ravel())
            for flat_idx in flat_order:
                ti, pi = divmod(int(flat_idx), len(peak_z))
                if ti in used_tracks or pi in used_peaks:
                    continue
                if dists[ti, pi] > match_tolerance:
                    break  # all remaining are farther
                tracks[ti]['x'].append(x)
                tracks[ti]['z'].append(peak_z[pi])
                tracks[ti]['score'].append(peak_scores[pi])
                active_track_z[ti] = peak_z[pi]
                used_tracks.add(ti)
                used_peaks.add(pi)

        # Start new tracks for unmatched peaks
        for pi in range(len(peak_z)):
            if pi not in used_peaks:
                tracks.append({
                    'x': [x],
                    'z': [peak_z[pi]],
                    'score': [peak_scores[pi]],
                })
                active_track_z.append(peak_z[pi])

    return tracks


def filter_tracks(tracks, min_length):
    """Remove tracks shorter than min_length detections."""
    return [t for t in tracks if len(t['x']) >= min_length]


def classify_tracks(tracks):
    """Classify tracks as 'joint' or 'crack' based on block height spacing.

    Uses pairwise spacing between adjacent tracks (sorted by mean Z)
    rather than a global grid, so it handles large settlement shifts
    (e.g. 3 ft over the wall length) where joints at one end are much
    lower than at the other.

    Logic: sort tracks by mean Z, check if each pair of adjacent tracks
    is spaced at approximately N * BLOCK_HEIGHT_M. Tracks that fit the
    spacing pattern with their neighbors are joints; isolated tracks
    with non-block-height gaps on both sides are cracks.

    Args:
        tracks: list of track dicts

    Returns:
        classified: list of track dicts with added 'label' key
            ('joint' or 'crack') and 'mean_z' key
    """
    if not tracks:
        return []

    expected = BLOCK_HEIGHT_M
    tolerance = BLOCK_HEIGHT_TOLERANCE

    # Compute mean Z for each track
    for t in tracks:
        t['mean_z'] = np.mean(t['z'])
    tracks_sorted = sorted(tracks, key=lambda t: t['mean_z'])

    def spacing_matches_block(z1, z2):
        """Check if the gap between two Z values is ~N * block height."""
        gap = abs(z2 - z1)
        if expected <= 0:
            return True
        n_blocks = round(gap / expected)
        if n_blocks == 0:
            return False
        residual = abs(gap - n_blocks * expected)
        return residual <= expected * tolerance

    # For each track, check if it has at least one neighbor at block spacing
    n = len(tracks_sorted)
    for i, t in enumerate(tracks_sorted):
        has_block_neighbor = False
        if i > 0 and spacing_matches_block(
                tracks_sorted[i - 1]['mean_z'], t['mean_z']):
            has_block_neighbor = True
        if i < n - 1 and spacing_matches_block(
                t['mean_z'], tracks_sorted[i + 1]['mean_z']):
            has_block_neighbor = True
        t['label'] = 'joint' if has_block_neighbor else 'crack'

    return tracks_sorted


# ── Stage 4: Spline fitting per track ────────────────────────────────────────

def fit_track_splines(tracks, smoothing, n_points):
    """Fit a cubic spline to each joint track.

    Args:
        tracks: list of track dicts from track_joints()
        smoothing: splprep smoothing factor
        n_points: number of points to evaluate along each spline

    Returns:
        splines: list of dicts with keys:
            'x': evaluated X positions (n_points,)
            'z': evaluated Z positions (n_points,)
            'tck': spline coefficients (for later evaluation)
    """
    splines = []
    for track in tracks:
        x = np.array(track['x'])
        z = np.array(track['z'])

        if len(x) < 4:
            # Not enough points for cubic spline, store raw
            splines.append({'x': x, 'z': z, 'tck': None})
            continue

        try:
            tck, u = splprep([x, z], s=smoothing)
            u_new = np.linspace(0, 1, n_points)
            x_smooth, z_smooth = splev(u_new, tck)
            splines.append({'x': x_smooth, 'z': z_smooth, 'tck': tck})
        except Exception:
            splines.append({'x': x, 'z': z, 'tck': None})

    return splines


# ── Full pipeline ────────────────────────────────────────────────────────────

def detect_horizontal_joints(points):
    """Run the full horizontal joint detection pipeline.

    Args:
        points: (N, 3) unrolled wall point cloud

    Returns:
        result: dict with keys:
            'raster_hz': horizontal joint score raster
            'raster_vt': vertical joint score raster
            'x_edges', 'z_edges': raster bin edges
            'tracks': raw joint tracks
            'splines': fitted spline curves
            'normals': (N, 3) estimated normals
    """
    resolution = JOINT_RASTER_RESOLUTION

    printf("Computing normals (knn=%d)..." % JOINT_NORMAL_KNN)
    normals = compute_normals(points, JOINT_NORMAL_KNN)

    printf("Rasterizing joint scores (resolution=%.3fm)..." % resolution)
    raster_hz, raster_vt, x_edges, z_edges = rasterize_joint_score(
        points, normals, resolution
    )

    printf("Detecting peaks in sliding windows (block height=%.0fin / %.3fm)..." % (
        BLOCK_HEIGHT_IN, BLOCK_HEIGHT_M))
    detections = detect_joints_windowed(
        raster_hz, x_edges, z_edges,
        window_width=JOINT_WINDOW_WIDTH,
        window_step=JOINT_WINDOW_STEP,
        sigma=JOINT_GAUSSIAN_SIGMA,
        min_height=JOINT_PEAK_MIN_HEIGHT,
        resolution=resolution,
    )
    printf("  %d windows, %d total detections" % (
        len(detections), sum(len(d[1]) for d in detections)))

    printf("Tracking joints across windows...")
    tracks = track_joints(detections, JOINT_MATCH_TOLERANCE)
    tracks = filter_tracks(tracks, JOINT_MIN_TRACK_LENGTH)
    printf("  %d tracks after filtering (min_length=%d)" % (
        len(tracks), JOINT_MIN_TRACK_LENGTH))

    printf("Classifying tracks (joint vs crack)...")
    tracks = classify_tracks(tracks)
    n_joints = sum(1 for t in tracks if t['label'] == 'joint')
    n_cracks = sum(1 for t in tracks if t['label'] == 'crack')
    printf("  %d joints, %d cracks (block grid tolerance=%.0f%%)" % (
        n_joints, n_cracks, BLOCK_HEIGHT_TOLERANCE * 100))

    printf("Fitting splines (joints only)...")
    joint_tracks = [t for t in tracks if t['label'] == 'joint']
    splines = fit_track_splines(joint_tracks, JOINT_SPLINE_SMOOTHING, JOINT_SPLINE_POINTS)

    return {
        'raster_hz': raster_hz,
        'raster_vt': raster_vt,
        'x_edges': x_edges,
        'z_edges': z_edges,
        'tracks': tracks,
        'joint_tracks': joint_tracks,
        'crack_tracks': [t for t in tracks if t['label'] == 'crack'],
        'splines': splines,
        'normals': normals,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        print("No displacement PLY files found.")
        sys.exit(1)

    for file in files:
        wall_id = os.path.basename(file).split("_")[1]
        printf("Processing wall %s: %s" % (wall_id, file))

        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.positions.numpy()
        printf("  %d points, X=[%.1f, %.1f], Z=[%.1f, %.1f]" % (
            len(points),
            points[:, 0].min(), points[:, 0].max(),
            points[:, 2].min(), points[:, 2].max(),
        ))

        result = detect_horizontal_joints(points)

        # Save spline point clouds
        out_dir = "outputs/point_clouds/unrolled"
        ensure_dir(out_dir + "/")
        for i, spline in enumerate(result['splines']):
            pts = np.zeros((len(spline['x']), 3))
            pts[:, 0] = spline['x']
            pts[:, 2] = spline['z']
            pcd = o3d.t.geometry.PointCloud()
            pcd.point.positions = pts
            colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(pts), 1))
            pcd.point.colors = colors
            path = "%s/joint_%s_%02d.ply" % (out_dir, wall_id, i)
            o3d.t.io.write_point_cloud(path, pcd)

        printf("  Saved %d joint splines for wall %s" % (
            len(result['splines']), wall_id))

    printf("Done.")
