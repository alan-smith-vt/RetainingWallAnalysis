"""
Core retaining wall analysis engine.

Processes 3D point cloud data to compute structural displacement and slope
characteristics of concrete retaining walls. For each wall, the script:
  1. Loads the main point cloud and wall vertex centerline.
  2. Rotates each wall segment to align with the x-axis.
  3. Extracts perpendicular cross-sections.
  4. Fits piecewise linear slopes in configurable segments.
  5. Colors points by slope magnitude and vertical displacement.
  6. Outputs colored point clouds, unrolled views, and slope CSVs.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import math
from config import (
    POINT_CLOUD_FILE, WALL_VERTICES_PATTERN, WALL_IDS, ANALYSIS_SPACINGS,
    SEGMENT_LENGTH, SLICE_HALF_WIDTH, TOP_OF_WALL_OFFSET,
    MAX_DISPLACEMENT_FOR_COLORS, SLOPE_THRESHOLD, SLOPE_COLORMAP_RANGE,
    TOP_INCHES_FOR_NEW_SLOPE, DISCRETE_SLOPE_RANGES,
)

print("[%s]: Python started." % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def savePoints(points, filePath, colors=None):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = points
    if colors is not None:
        pcd.point.colors = colors
    o3d.t.io.write_point_cloud(filePath, pcd)


def getCrossSection(slicePoints, linePoints, filePath, k, slopeColor):
    points = np.vstack([slicePoints, linePoints])
    points[:, 1] = points[:, 1] - np.min(points[:, 1]) - k
    colors_1 = np.ones_like(slicePoints) * np.array([255, 255, 255])
    colors_2 = np.ones_like(linePoints) * np.array(slopeColor)
    colors = np.vstack([colors_1, colors_2])
    return points, colors


def fixSpacing(points, spacing=1):
    points = np.vstack([points, points[-1, :]])
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    total_length = distances[-1]
    num_points = int(total_length / spacing) + 1
    new_distances = np.linspace(0, total_length, num_points)
    new_vertices = np.zeros((num_points, 3))
    for i in range(3):
        new_vertices[:, i] = np.interp(new_distances, distances, points[:, i])
    return new_vertices


def hex_to_rgb(hex_color):
    """Convert hex color to RGB array."""
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def value_to_rgb_discrete(percent):
    """Map a float value to unique hex colors based on percentage ranges."""
    percent = -percent
    for low, high, color in DISCRETE_SLOPE_RANGES:
        if low <= percent < high or (high == 0.01 and percent <= high):
            return hex_to_rgb(color)


def value_to_rgb_jet(percent):
    if percent > 0:
        return hex_to_rgb('#000000')
    cmap = plt.cm.jet
    percent = (-percent * 100) / SLOPE_COLORMAP_RANGE
    return cmap(percent)[:3]


def value_to_rgb(percent, thresh):
    if thresh is None:
        return value_to_rgb_jet(percent)
    if percent < thresh:
        percent = thresh
        cmap = plt.cm.jet
        percent = (-percent * 100) / SLOPE_COLORMAP_RANGE
        return cmap(percent)[:3]
    else:
        return hex_to_rgb('#ffffff')


def fitLineZ1toZ2(points_2d, z1, z2, x1, x2, thresh):
    z_values = points_2d[:, 1]

    j1 = np.searchsorted(z_values, z1)
    j2 = np.searchsorted(z_values, z2)

    sub_points_2d = points_2d[j1:j2, :]
    if len(sub_points_2d) < 10:
        return None, None, None, None, None

    y_coords = sub_points_2d[:, 0]
    z_coords = sub_points_2d[:, 1]

    A = np.vstack([z_coords, np.ones(len(z_coords))]).T
    slope, intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]

    z = np.linspace(z1, z2, 100)
    y = slope * z + intercept

    x_vals = np.linspace(x1, x2, 100)
    lines = []
    line_colors_list = []

    for x_val in x_vals[1:]:
        x = np.ones_like(z) * x_val
        line = np.column_stack([x, y, z])
        lines.append(line)

        slope_color = value_to_rgb(slope, thresh)
        line_colors = np.tile(slope_color, (line.shape[0], 1))
        line_colors_list.append(line_colors)

    all_lines = np.vstack(lines)
    all_line_colors = np.vstack(line_colors_list)

    return slope, intercept, y, all_lines, all_line_colors


def unrollSlices(pc_slices, spacing):
    for i in range(len(pc_slices)):
        axis = 0
        x_min = np.min(pc_slices[i][:, axis])
        pc_slices[i][:, axis] = (pc_slices[i][:, axis] - x_min) + spacing * i
    return np.vstack(pc_slices)


# ── Main execution ──────────────────────────────────────────────────────────

points_file = POINT_CLOUD_FILE
thresh = SLOPE_THRESHOLD

for spacing in ANALYSIS_SPACINGS:
    print("~" * 50)
    printf("Processing retaining walls with %.1f slice spacing" % spacing)
    print("~" * 50)
    for wall_id in WALL_IDS:
        print("~" * 50)
        printf("Starting Wall %d" % wall_id)
        print("~" * 50)
        vertices_file = WALL_VERTICES_PATTERN.format(wall_id=wall_id)

        pc_source = o3d.t.io.read_point_cloud(points_file)
        points = pc_source.point.positions.numpy()

        vertices_source = o3d.t.io.read_point_cloud(vertices_file)
        vertices = vertices_source.point.positions.numpy()
        vertices = fixSpacing(vertices, spacing=spacing)

        lines = []
        line_colors = []
        pc_slices = []
        pc_slice_colors = []
        slope_values = []
        pc_slope_colors = []
        pc_slices_rotated = []
        lines_rotated = []
        new_slopes = []
        new_slope_colors = []

        segLength = SEGMENT_LENGTH

        points_source = points.copy()

        # Pre-filter points once to avoid repeated filtering
        global_z_min = np.min(vertices[:, 2]) - 1.0
        points_master = points_source[points_source[:, 2] > global_z_min, :]

        for i in tqdm(range(len(vertices) - 1)):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            z_min = np.min([vertices[i, 2], vertices[i + 1, 2]])
            z_mask = points_master[:, 2] > z_min
            points = points_master[z_mask, :]

            direction = v2 - v1
            angle = -np.arctan2(direction[1], direction[0])

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                        [sin_a, cos_a, 0],
                                        [0, 0, 1]])

            cos_a_inv, sin_a_inv = np.cos(-angle), np.sin(-angle)
            inverse_rotation_matrix = np.array([[cos_a_inv, -sin_a_inv, 0],
                                                [sin_a_inv, cos_a_inv, 0],
                                                [0, 0, 1]])

            points_rotated = np.dot(points - v1, rotation_matrix.T) + v1
            vertices_rotated = np.dot(vertices - v1, rotation_matrix.T) + v1

            v1_rotated = vertices_rotated[i]
            v2_rotated = vertices_rotated[i + 1]

            x_values = points_rotated[:, 0]
            x_min, x_max = min(v1_rotated[0], v2_rotated[0]), max(v1_rotated[0], v2_rotated[0])

            x_mask = (x_values >= x_min) & (x_values <= x_max)
            pc_slice = points_rotated[x_mask]

            y_mask = np.abs(pc_slice[:, 1] - v1_rotated[1]) <= SLICE_HALF_WIDTH
            pc_slice = pc_slice[y_mask]

            points_2d = pc_slice[:, 1:3]
            sorted_indices = np.argsort(points_2d[:, 1])
            points_2d = points_2d[sorted_indices]

            z_max = np.max(pc_slice[:, 2]) + TOP_OF_WALL_OFFSET
            x_val = (v1_rotated[0] + v2_rotated[0]) / 2

            numSlopes = max(int((z_max - z_min) / segLength), 1)

            lines_rotated_temp = []
            line_colors_temp = []
            for j in range(numSlopes):
                z1 = (j / numSlopes) * (z_max - z_min) + z_min
                z2 = ((j + 1) / numSlopes) * (z_max - z_min) + z_min

                slope, intercept, y, line, line_color = fitLineZ1toZ2(points_2d, z1, z2, v1_rotated[0], v2_rotated[0], thresh)
                if slope is None:
                    continue
                line_unrotated = np.dot(line - v1, inverse_rotation_matrix.T) + v1
                lines.append(line_unrotated)
                line_colors_temp.append(line_color)
                lines_rotated_temp.append(line)

            if len(lines_rotated_temp) > 0:
                lines_rotated.append(np.vstack(lines_rotated_temp))
                line_colors.append(np.vstack(line_colors_temp))

            slope, intercept, y, line, line_color = fitLineZ1toZ2(points_2d, z_min, z_max, v1_rotated[0], v2_rotated[0], thresh)

            if slope is None:
                continue
            y_ref = slope * z_min + intercept

            # New slope algorithm
            top18_ind = np.searchsorted(points_2d[:, 1], z_max - TOP_INCHES_FOR_NEW_SLOPE)
            avg_y = np.mean(points_2d[top18_ind:, 0])
            delta_y = (avg_y - y_ref)
            delta_z = z_max - z_min
            new_slope = (delta_y / delta_z)
            new_slopes.append(new_slope)
            new_slope_color = np.tile(value_to_rgb_jet(new_slope), (line.shape[0], 1))
            new_slope_colors.append(new_slope_color)

            cmap = plt.cm.jet

            slope_values.append(slope)

            pc_slices_rotated.append(pc_slice)

            pc_slice_unrotated = np.dot(pc_slice - v1, inverse_rotation_matrix.T) + v1
            line_unrotated = np.dot(line - v1, inverse_rotation_matrix.T) + v1

            pc_slice_deltas = -(pc_slice[:, 1] - y_ref) / MAX_DISPLACEMENT_FOR_COLORS
            colors_pc_slice = cmap(pc_slice_deltas)[:, :3]

            pc_slices.append(pc_slice_unrotated)
            pc_slice_colors.append(colors_pc_slice)

        pc_slices_unrolled = unrollSlices(pc_slices_rotated, spacing)
        lines_unrolled = unrollSlices(lines_rotated, spacing)

        pc_slices = np.vstack(pc_slices)
        pc_slice_colors = np.vstack(pc_slice_colors)
        lines = np.vstack(lines)
        line_colors = np.vstack(line_colors)
        slope_values = np.vstack(slope_values)

        new_slope_colors = np.vstack(new_slope_colors)

        points_temp = []
        colors_temp = []
        for k in range(len(pc_slices_rotated)):
            points, colors = getCrossSection(pc_slices_rotated[k], lines_rotated[k], "pointClouds/crossSections/wall_%d/section_%d.ply" % (wall_id, k), k, value_to_rgb_jet(slope_values[k][0])[:3])
            points_temp.append(points)
            colors_temp.append(colors)
        points = np.vstack(points_temp)
        colors = np.vstack(colors_temp)
        savePoints(points, "pointClouds/unrolled/crossSections/wall_%d.ply" % wall_id, colors=colors)

        if thresh is None:
            savePoints(pc_slices, "pointClouds/displacements/pc_slice_%d_%.1f.ply" % (wall_id, spacing), colors=pc_slice_colors)
            savePoints(lines, "pointClouds/slopes/line_%d_%.1f.ply" % (wall_id, spacing), colors=line_colors)
            np.savetxt("renders/slopes/slope_%d_%.1f.csv" % (wall_id, spacing), slope_values, delimiter=",", fmt='%.6f')
            savePoints(lines, "pointClouds/new_slopes/line_%d_%.1f.ply" % (wall_id, spacing), colors=new_slope_colors)

            savePoints(pc_slices_unrolled, "pointClouds/unrolled/displacements/pc_slices_unrolled_%d_%.1f.ply" % (wall_id, spacing), colors=pc_slice_colors)
            savePoints(lines_unrolled, "pointClouds/unrolled/slopes/lines_unrolled_%d_%.1f.ply" % (wall_id, spacing), colors=line_colors)
            savePoints(lines_unrolled, "pointClouds/unrolled/new_slopes/line_%d_%.1f.ply" % (wall_id, spacing), colors=new_slope_colors)
        else:
            savePoints(lines_unrolled, "pointClouds/unrolled/slopes/lines_unrolled_%d_%3.3f_%.1f.ply" % (wall_id, thresh, spacing), colors=line_colors)
