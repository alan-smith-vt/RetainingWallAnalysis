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

Cross-section slices are processed in parallel using multiprocessing with
shared memory for the point cloud array.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from multiprocessing import shared_memory, Pool
import math


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

from config import (
    POINT_CLOUD_FILE, WALL_VERTICES_PATTERN, WALL_IDS, ANALYSIS_SPACINGS,
    SEGMENT_LENGTH, SEGMENT_OVERLAP, SLICE_OVERLAP, SLICE_HALF_WIDTH,
    TOP_OF_WALL_OFFSET, MAX_DISPLACEMENT_FOR_COLORS, EXPECTED_WALL_SLOPE,
    SLOPE_THRESHOLD, SLOPE_COLORMAP_RANGE,
    TOP_INCHES_FOR_NEW_SLOPE, DISCRETE_SLOPE_RANGES,
    NUM_WORKERS,
)


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def savePoints(points, filePath, colors=None, scalars=None):
    ensure_dir(filePath)
    if scalars:
        _write_ply_with_scalars(points, filePath, colors, scalars)
    else:
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = points
        if colors is not None:
            pcd.point.colors = colors
        o3d.t.io.write_point_cloud(filePath, pcd)


def _write_ply_with_scalars(points, filePath, colors, scalars):
    """Write PLY with scalar fields as standard properties CloudCompare can read."""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)

    header = "ply\nformat binary_little_endian 1.0\n"
    header += "element vertex %d\n" % n
    header += "property float x\nproperty float y\nproperty float z\n"

    has_colors = colors is not None
    if has_colors:
        colors = np.asarray(colors)
        # Convert 0-1 float colors to 0-255 uint8 if needed
        if colors.dtype in (np.float32, np.float64):
            if colors.max() <= 1.0:
                colors = (colors * 255).clip(0, 255).astype(np.uint8)
            else:
                colors = colors.clip(0, 255).astype(np.uint8)
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"

    # Write scalar value as intensity (universally recognized by point cloud viewers)
    scalar_key = next(iter(scalars))
    scalar_array = np.asarray(scalars[scalar_key], dtype=np.float32).ravel()
    header += "property float intensity\n"

    header += "end_header\n"

    with open(filePath, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(points[i].tobytes())
            if has_colors:
                f.write(colors[i].tobytes())
            f.write(scalar_array[i:i+1].tobytes())


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
    # Centered jet: more batter than expected = blue, less = red, on-profile = green/yellow
    deviation = percent - EXPECTED_WALL_SLOPE
    cmap = plt.cm.jet
    mapped = np.clip((-deviation * 100) / SLOPE_COLORMAP_RANGE * 0.5 + 0.5, 0, 1)
    return cmap(mapped)[:3]


def value_to_rgb(percent, thresh):
    if thresh is None:
        return value_to_rgb_jet(percent)
    deviation = percent - EXPECTED_WALL_SLOPE
    if abs(deviation) < abs(thresh):
        return hex_to_rgb('#ffffff')
    return value_to_rgb_jet(percent)


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


# ── Shared memory helpers ───────────────────────────────────────────────────

# Module-level references set by _init_worker
_shm = None
_points_master = None
_pm_shape = None
_pm_dtype = None


def _init_worker(shm_name, shape, dtype):
    """Attach to the shared memory block in each worker process."""
    global _shm, _points_master, _pm_shape, _pm_dtype
    _pm_shape = shape
    _pm_dtype = dtype
    _shm = shared_memory.SharedMemory(name=shm_name)
    _points_master = np.ndarray(shape, dtype=dtype, buffer=_shm.buf)


def process_slice(args):
    """
    Process a single cross-section slice. Runs in a worker process.

    Reads points_master from shared memory. Returns a dict with all
    per-slice outputs, or None if the slice is empty.
    """
    i, v1, v2, vertices, thresh, segLength, spacing, seg_overlap, slice_overlap = args

    # Access shared points_master
    points_master = _points_master

    z_min = min(v1[2], v2[2])
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

    if len(pc_slice) == 0:
        return None

    points_2d = pc_slice[:, 1:3]
    sorted_indices = np.argsort(points_2d[:, 1])
    points_2d = points_2d[sorted_indices]

    z_max = np.max(pc_slice[:, 2]) + TOP_OF_WALL_OFFSET

    # Horizontal sub-steps for slope analysis
    slice_width = x_max - x_min
    h_step = max(slice_width - slice_overlap, 0.01)
    n_h_steps = max(math.ceil(slice_width / h_step), 1)

    # Piecewise slope fitting with vertical and horizontal overlap
    v_step = max(segLength - seg_overlap, 0.1)
    wall_height = z_max - z_min
    numSlopes = max(math.ceil(wall_height / v_step), 1)

    piecewise_lines_unrotated = []
    piecewise_line_colors = []
    piecewise_lines_rotated = []
    piecewise_line_scalars = []
    for h in range(n_h_steps):
        # Horizontal sub-step: draw region
        draw_x1 = x_min + h * h_step
        draw_x2 = min(draw_x1 + h_step, x_max)

        # Fit using all points in the full slice width (analysis_window)
        for j in range(numSlopes):
            # Vertical: draw region
            draw_z1 = z_min + j * v_step
            draw_z2 = min(draw_z1 + v_step, z_max)

            # Expanded vertical data window for fitting
            fit_z1 = max(draw_z1 - seg_overlap / 2, z_min)
            fit_z2 = min(draw_z2 + seg_overlap / 2, z_max)

            slope_val, intercept, y, line, line_color = fitLineZ1toZ2(points_2d, fit_z1, fit_z2, draw_x1, draw_x2, thresh)
            if slope_val is None:
                continue

            # Generate line geometry over the draw region only
            draw_height = draw_z2 - draw_z1
            draw_width = abs(draw_x2 - draw_x1)
            n_z = max(int(15 * draw_height), 2)
            n_x = max(int(30 * draw_width), 2)
            z_draw = np.linspace(draw_z1, draw_z2, n_z)
            y_draw = slope_val * z_draw + intercept
            x_vals = np.linspace(draw_x1, draw_x2, n_x)
            draw_lines = []
            draw_colors = []
            slope_color = value_to_rgb(slope_val, thresh)
            for x_val in x_vals[1:]:
                x = np.ones_like(z_draw) * x_val
                seg = np.column_stack([x, y_draw, z_draw])
                draw_lines.append(seg)
                draw_colors.append(np.tile(slope_color, (seg.shape[0], 1)))

            line_seg = np.vstack(draw_lines)
            line_seg_colors = np.vstack(draw_colors)
            line_seg_scalars = np.full(len(line_seg), slope_val, dtype=np.float32)

            line_unrotated = np.dot(line_seg - v1, inverse_rotation_matrix.T) + v1
            piecewise_lines_unrotated.append(line_unrotated)
            piecewise_line_colors.append(line_seg_colors)
            piecewise_lines_rotated.append(line_seg)
            piecewise_line_scalars.append(line_seg_scalars)

    if len(piecewise_lines_rotated) == 0:
        lines_rotated_combined = None
        line_colors_combined = None
        line_scalars_combined = None
    else:
        lines_rotated_combined = np.vstack(piecewise_lines_rotated)
        line_colors_combined = np.vstack(piecewise_line_colors)
        line_scalars_combined = np.concatenate(piecewise_line_scalars)

    # Full-height fit
    slope, intercept, y, line, line_color = fitLineZ1toZ2(points_2d, z_min, z_max, v1_rotated[0], v2_rotated[0], thresh)

    if slope is None:
        return None
    y_ref = slope * z_min + intercept

    # Expected slope reference line
    exp_z = np.linspace(z_min, z_max, 100)
    exp_y = y_ref + EXPECTED_WALL_SLOPE * (exp_z - z_min)
    exp_x_vals = np.linspace(v1_rotated[0], v2_rotated[0], 100)
    exp_lines_temp = []
    for x_val in exp_x_vals[1:]:
        exp_x = np.ones_like(exp_z) * x_val
        exp_lines_temp.append(np.column_stack([exp_x, exp_y, exp_z]))
    exp_line_rotated = np.vstack(exp_lines_temp)
    exp_line_unrotated = np.dot(exp_line_rotated - v1, inverse_rotation_matrix.T) + v1

    # New slope algorithm — measure deviation from expected batter
    top18_ind = np.searchsorted(points_2d[:, 1], z_max - TOP_INCHES_FOR_NEW_SLOPE)
    avg_y = np.mean(points_2d[top18_ind:, 0])
    y_expected_top = y_ref + EXPECTED_WALL_SLOPE * (z_max - z_min)
    delta_y = (avg_y - y_expected_top)
    delta_z = z_max - z_min
    new_slope = (delta_y / delta_z)

    # new_slope is already a deviation — map directly to centered jet
    ns_mapped = np.clip((-new_slope * 100) / SLOPE_COLORMAP_RANGE * 0.5 + 0.5, 0, 1)
    new_slope_color = np.tile(plt.cm.jet(ns_mapped)[:3], (line.shape[0], 1))
    line_unrotated_full = np.dot(line - v1, inverse_rotation_matrix.T) + v1

    cmap = plt.cm.jet

    pc_slice_unrotated = np.dot(pc_slice - v1, inverse_rotation_matrix.T) + v1

    # Displacement coloring: centered jet on expected batter
    y_expected = y_ref + EXPECTED_WALL_SLOPE * (pc_slice[:, 2] - z_min)
    pc_slice_displacement = pc_slice[:, 1] - y_expected  # raw displacement in meters
    pc_slice_deltas = -pc_slice_displacement / MAX_DISPLACEMENT_FOR_COLORS
    pc_slice_deltas_mapped = np.clip(pc_slice_deltas * 0.5 + 0.5, 0, 1)
    colors_pc_slice = cmap(pc_slice_deltas_mapped)[:, :3]

    # Scalar for new_slope line: constant deviation value per point
    new_slope_scalar = np.full(len(line), new_slope, dtype=np.float32)

    return {
        'i': i,
        'slope': slope,
        'new_slope': new_slope,
        # Piecewise slope data
        'lines_unrotated': piecewise_lines_unrotated,
        'lines_rotated': lines_rotated_combined,
        'line_colors': line_colors_combined,
        'line_scalars': line_scalars_combined,
        # New slope data
        'new_slope_color': new_slope_color,
        'new_slope_scalar': new_slope_scalar,
        'new_slope_line': line_unrotated_full,
        'new_slope_line_rotated': line,
        # Expected slope reference line
        'expected_slope_line': exp_line_unrotated,
        'expected_slope_line_rotated': exp_line_rotated,
        # Point cloud slice data
        'pc_slice_rotated': pc_slice,
        'pc_slice_unrotated': pc_slice_unrotated,
        'pc_slice_colors': colors_pc_slice,
        'pc_slice_displacement': pc_slice_displacement,
    }


# ── Main execution ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("[%s]: Python started." % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    points_file = POINT_CLOUD_FILE
    thresh = SLOPE_THRESHOLD

    num_workers = NUM_WORKERS
    if num_workers == 0:
        num_workers = os.cpu_count() or 4
    parallel = num_workers > 1

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

            points_source = points.copy()

            # Pre-filter points once
            global_z_min = np.min(vertices[:, 2]) - 1.0
            points_master = points_source[points_source[:, 2] > global_z_min, :]

            # Build argument list for each slice
            n_slices = len(vertices) - 1
            args_list = [
                (i, vertices[i].copy(), vertices[i + 1].copy(), vertices.copy(), thresh, SEGMENT_LENGTH, spacing, SEGMENT_OVERLAP, SLICE_OVERLAP)
                for i in range(n_slices)
            ]

            # Create shared memory for points_master
            shm = shared_memory.SharedMemory(create=True, size=points_master.nbytes)
            shm_array = np.ndarray(points_master.shape, dtype=points_master.dtype, buffer=shm.buf)
            np.copyto(shm_array, points_master)

            printf("Processing %d slices with %d workers" % (n_slices, num_workers))

            if parallel:
                pool = Pool(
                    processes=num_workers,
                    initializer=_init_worker,
                    initargs=(shm.name, points_master.shape, points_master.dtype),
                )
                results_raw = list(tqdm(
                    pool.imap(process_slice, args_list),
                    total=n_slices,
                ))
                pool.close()
                pool.join()
            else:
                # Serial mode — initialize the worker globals in this process
                _init_worker(shm.name, points_master.shape, points_master.dtype)
                results_raw = []
                for args in tqdm(args_list):
                    results_raw.append(process_slice(args))

            # Clean up shared memory
            shm.close()
            shm.unlink()

            # Filter out None results (empty slices) and sort by index
            results = [r for r in results_raw if r is not None]
            results.sort(key=lambda r: r['i'])

            printf("Assembling %d valid slices" % len(results))

            # Assemble outputs from results
            lines = []
            line_colors = []
            line_scalars = []
            pc_slices = []
            pc_slice_colors = []
            pc_slice_displacements = []
            slope_values = []
            pc_slices_rotated = []
            lines_rotated = []
            new_slopes = []
            new_slope_colors = []
            new_slope_scalars = []
            new_slope_lines = []
            new_slope_lines_rotated = []
            expected_slope_lines = []
            expected_slope_lines_rotated = []

            for r in results:
                slope_values.append(r['slope'])
                new_slopes.append(r['new_slope'])

                for lu in r['lines_unrotated']:
                    lines.append(lu)

                if r['lines_rotated'] is not None:
                    lines_rotated.append(r['lines_rotated'])
                    line_colors.append(r['line_colors'])
                    line_scalars.append(r['line_scalars'])

                new_slope_colors.append(r['new_slope_color'])
                new_slope_scalars.append(r['new_slope_scalar'])
                new_slope_lines.append(r['new_slope_line'])
                new_slope_lines_rotated.append(r['new_slope_line_rotated'])

                expected_slope_lines.append(r['expected_slope_line'])
                expected_slope_lines_rotated.append(r['expected_slope_line_rotated'])

                pc_slices_rotated.append(r['pc_slice_rotated'])
                pc_slices.append(r['pc_slice_unrotated'])
                pc_slice_colors.append(r['pc_slice_colors'])
                pc_slice_displacements.append(r['pc_slice_displacement'])

            pc_slices_unrolled = unrollSlices([s.copy() for s in pc_slices_rotated], spacing)
            lines_unrolled = unrollSlices([lr.copy() for lr in lines_rotated], spacing)

            pc_slices = np.vstack(pc_slices)
            pc_slice_colors = np.vstack(pc_slice_colors)
            pc_slice_displacements = np.concatenate(pc_slice_displacements)
            lines = np.vstack(lines)
            line_colors = np.vstack(line_colors)
            line_scalars = np.concatenate(line_scalars)
            slope_values = np.vstack(slope_values)

            new_slope_colors = np.vstack(new_slope_colors)
            new_slope_scalars = np.concatenate(new_slope_scalars)
            new_slope_lines_all = np.vstack(new_slope_lines)
            expected_slope_lines_all = np.vstack(expected_slope_lines)

            points_temp = []
            colors_temp = []
            for k in range(len(pc_slices_rotated)):
                pts, cols = getCrossSection(pc_slices_rotated[k], lines_rotated[k], "pointClouds/crossSections/wall_%d/section_%d.ply" % (wall_id, k), k, value_to_rgb_jet(slope_values[k][0])[:3])
                points_temp.append(pts)
                colors_temp.append(cols)
            pts = np.vstack(points_temp)
            cols = np.vstack(colors_temp)
            savePoints(pts, "pointClouds/unrolled/crossSections/wall_%d.ply" % wall_id, colors=cols)

            if thresh is None:
                savePoints(pc_slices, "pointClouds/displacements/pc_slice_%d_%.1f.ply" % (wall_id, spacing), colors=pc_slice_colors,
                           scalars={'displacement': pc_slice_displacements})
                savePoints(lines, "pointClouds/slopes/line_%d_%.1f.ply" % (wall_id, spacing), colors=line_colors,
                           scalars={'slope': line_scalars})
                ensure_dir("renders/slopes/slope_%d_%.1f.csv" % (wall_id, spacing))
                np.savetxt("renders/slopes/slope_%d_%.1f.csv" % (wall_id, spacing), slope_values, delimiter=",", fmt='%.6f')
                savePoints(new_slope_lines_all, "pointClouds/new_slopes/line_%d_%.1f.ply" % (wall_id, spacing), colors=new_slope_colors,
                           scalars={'slope_deviation': new_slope_scalars})
                savePoints(expected_slope_lines_all, "pointClouds/expected_slopes/line_%d_%.1f.ply" % (wall_id, spacing))

                savePoints(pc_slices_unrolled, "pointClouds/unrolled/displacements/pc_slices_unrolled_%d_%.1f.ply" % (wall_id, spacing), colors=pc_slice_colors,
                           scalars={'displacement': pc_slice_displacements})
                savePoints(lines_unrolled, "pointClouds/unrolled/slopes/lines_unrolled_%d_%.1f.ply" % (wall_id, spacing), colors=line_colors,
                           scalars={'slope': line_scalars})
                new_slope_lines_unrolled = unrollSlices([nsr.copy() for nsr in new_slope_lines_rotated], spacing)
                savePoints(new_slope_lines_unrolled, "pointClouds/unrolled/new_slopes/line_%d_%.1f.ply" % (wall_id, spacing), colors=new_slope_colors,
                           scalars={'slope_deviation': new_slope_scalars})
                expected_slope_lines_unrolled = unrollSlices([esr.copy() for esr in expected_slope_lines_rotated], spacing)
                savePoints(expected_slope_lines_unrolled, "pointClouds/unrolled/expected_slopes/line_%d_%.1f.ply" % (wall_id, spacing))
            else:
                savePoints(lines_unrolled, "pointClouds/unrolled/slopes/lines_unrolled_%d_%3.3f_%.1f.ply" % (wall_id, thresh, spacing), colors=line_colors,
                           scalars={'slope': line_scalars})

            printf("Wall %d complete" % wall_id)
