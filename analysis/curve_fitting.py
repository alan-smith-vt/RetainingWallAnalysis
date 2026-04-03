"""
Settlement curve fitting from density peaks.

Uses a sliding-window approach to find peak-density points along unrolled
elevation point clouds, then fits a smoothing spline to extract the
settlement profile. Outputs peak points, smoothed curves, and settlement
CSV files.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from io import BytesIO
import cv2
import re
from tqdm import tqdm
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from config import (
    CURVE_WINDOW_THICKNESS, CURVE_STEP_SIZE, CURVE_BIN_WIDTH,
    CURVE_CROP_DELTA, CURVE_OUTLIER_THRESHOLD,
    CURVE_SMOOTHING_FACTOR_1, CURVE_SMOOTHING_FACTOR_2,
    CURVE_NUM_SMOOTH_POINTS, ELEVATION_WINDOW_M, VERTICAL_EXAGGERATION,
    METERS_TO_FEET,
)

print("[%s]: Python started." % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def savePoints(points, filePath, color=None):
    ensure_dir(filePath)
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = points
    if color is not None:
        colors = np.tile(color, (len(points), 1))
        pcd.point.colors = colors
    o3d.t.io.write_point_cloud(filePath, pcd)


def extractTile_maxDensity(npPoints, bin_width, slice_axis, debug=False):
    sorted_indices = np.argsort(npPoints[:, slice_axis])
    sorted_points_raw = npPoints[sorted_indices]
    sorted_points = sorted_points_raw - sorted_points_raw[0, slice_axis]

    z_min = sorted_points[0, slice_axis]
    z_max = sorted_points[-1, slice_axis]
    z_range = z_max - z_min
    num_slices = int(z_range // bin_width)
    z_values = sorted_points[:, slice_axis]

    z_density = []
    prev_idx = 0
    for i in range(1, num_slices):
        target_z = bin_width * i
        next_idx = np.searchsorted(z_values, target_z)
        z_density.append(next_idx - prev_idx)
        prev_idx = next_idx

    z_density_array = np.array(z_density)
    x_axis = np.linspace(0, 1, num_slices) * z_range - z_min
    x_axis_density = x_axis[1:] - bin_width / 2

    if len(z_density_array) == 0:
        return None
    max_density = np.max(z_density_array)
    max_density_idx = z_density.index(max_density)

    loc_cm = x_axis_density[max_density_idx]
    crop_delta = CURVE_CROP_DELTA
    cm_above = loc_cm + crop_delta
    cm_below = loc_cm - crop_delta

    idx_above = np.searchsorted(x_axis_density, cm_above)
    idx_below = np.searchsorted(x_axis_density, cm_below)

    if idx_above >= len(x_axis_density):
        idx_above = len(x_axis_density) - 1

    if debug:
        printf("Counted points in each bin")
        plt.plot(x_axis_density, z_density_array)
        plt.scatter(loc_cm, max_density)
        plt.legend(["Density", "Max", "Chosen Splits"])
        plt.show()

    sorted_points = npPoints[sorted_indices]
    z_min = sorted_points[0, slice_axis]
    loc_cm = loc_cm + z_min

    return loc_cm


def sortAndSlice(npPoints, window_thickness, step_size, axis):
    sorted_indices = np.argsort(npPoints[:, axis])
    sorted_points = npPoints[sorted_indices]

    z_min = sorted_points[0, axis]
    z_max = sorted_points[-1, axis]
    z_values = sorted_points[:, axis]

    slice_points = []

    window_start = z_min

    while window_start <= z_max:
        window_end = window_start + window_thickness

        start_idx = np.searchsorted(z_values, window_start, side='left')
        end_idx = np.searchsorted(z_values, window_end, side='left')

        if start_idx < end_idx:
            slice_points.append(sorted_points[start_idx:end_idx])

        window_start += step_size

        if window_start > z_max:
            break

    return slice_points


# ── Main execution ──────────────────────────────────────────────────────────

files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
for file in files:
    # Extract wall_id from filename like displacement_1_0.1.ply
    wall_id = os.path.basename(file).split("_")[1]
    pc_source = o3d.t.io.read_point_cloud(file)
    points = pc_source.point.positions.numpy()
    x_min = np.min(points, axis=0)[0]
    window = CURVE_WINDOW_THICKNESS
    step = CURVE_STEP_SIZE
    slices = sortAndSlice(points, window, step, 0)
    peaks = []
    debug = False
    for i in tqdm(range(len(slices) - 1)):
        peak = extractTile_maxDensity(slices[i], CURVE_BIN_WIDTH, 2, debug=debug)
        if not peak:
            continue
        peaks.append(np.array([x_min + step * i, 0, peak]))
    peaks.append(np.array([x_min + step * len(slices) - 1, 0, peak]))
    peaks = np.vstack(peaks)
    savePoints(peaks, "outputs/point_clouds/unrolled/elevation_curve_%s_peaks.ply" % wall_id, color=np.array([255, 0, 255], dtype=np.uint8))

    x = peaks[:, 0]
    y = peaks[:, 2]
    tck, u = splprep([x, y], s=CURVE_SMOOTHING_FACTOR_1)

    # Second pass — remove outliers
    u_vals = x / np.max(x)
    _, y_smooth = splev(u_vals, tck)
    residuals = np.abs(y - y_smooth)
    inds = residuals > CURVE_OUTLIER_THRESHOLD
    x = x[~inds]
    y = y[~inds]

    tck, u = splprep([x, y], s=CURVE_SMOOTHING_FACTOR_2)

    u_new = np.linspace(0, 1, CURVE_NUM_SMOOTH_POINTS)
    x_smooth, y_smooth = splev(u_new, tck)
    peaks_smooth = np.zeros((len(x_smooth), 3))
    peaks_smooth[:, 0] = x_smooth
    peaks_smooth[:, 2] = y_smooth

    savePoints(peaks_smooth, "outputs/point_clouds/unrolled/elevation_curve_%s.ply" % wall_id, color=np.array([0, 0, 255], dtype=np.uint8))

    x_sample = np.arange(0, np.max(x), 1)
    x_sample = x_sample[x_sample > np.min(x)]
    x_sample = np.insert(x_sample, 0, np.min(x))
    x_sample = np.append(x_sample, np.max(x))

    u_sample = x_sample / np.max(x)
    _, y_sample = splev(u_sample, tck)

    delta_z = ELEVATION_WINDOW_M
    scale = VERTICAL_EXAGGERATION
    y_sample = (y_sample / (scale * delta_z) + 760.5)

    res = np.column_stack([x_sample * METERS_TO_FEET, y_sample])

    ensure_dir("excel_data/wall_%s_settlement.csv" % wall_id)
    np.savetxt("excel_data/wall_%s_settlement.csv" % wall_id, res, delimiter=",")
