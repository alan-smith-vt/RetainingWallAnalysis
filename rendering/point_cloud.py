"""
Point cloud to 2D image renderer.

Converts colored 3D point clouds into 2D scatter-plot PNG images using
matplotlib, with configurable marker sizes per target type.
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
from config import (
    RENDER_DPI, RENDER_RESOLUTION, RENDER_TARGET,
    MARKER_SIZE_DEFAULT, MARKER_SIZE_DISPLACEMENTS, MARKER_SIZE_SLOPES,
    EXPECTED_WALL_SLOPE, SLOPE_COLORMAP_RANGE, MAX_DISPLACEMENT_FOR_COLORS,
    FEET_TO_METERS,
)
try:
    from config import MAX_DISPLACEMENT_POSITIVE, MAX_DISPLACEMENT_NEGATIVE
except ImportError:
    MAX_DISPLACEMENT_POSITIVE = None
    MAX_DISPLACEMENT_NEGATIVE = None
try:
    from config import SLOPE_RANGE_POSITIVE, SLOPE_RANGE_NEGATIVE
except ImportError:
    SLOPE_RANGE_POSITIVE = None
    SLOPE_RANGE_NEGATIVE = None
try:
    from config import STATION_SPLITS
except ImportError:
    STATION_SPLITS = None
try:
    from config import STATION_MAX_FT, STATION_START_OFFSET_IN, STATION_END_OFFSET_IN
except ImportError:
    STATION_MAX_FT = None
    STATION_START_OFFSET_IN = 0
    STATION_END_OFFSET_IN = 0

print("[%s]: Python started." % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def zero_pc(pc, ref_pc=None):
    points = pc.point.positions.numpy()
    if ref_pc is None:
        points = points - np.min(points, axis=0)
    else:
        ref_points = ref_pc.point.positions.numpy()
        points = points - np.min(ref_points, axis=0)
    colors = pc.point.colors.numpy()

    pc_zeroed = o3d.t.geometry.PointCloud()
    pc_zeroed.point.positions = points
    pc_zeroed.point.colors = colors
    return pc_zeroed


def projectToImage1000_color(points, xyExtents, colors, x_axis, y_axis, z_axis, sz=72.):
    fig = plt.figure()
    ax2 = plt.gca()
    fig.dpi = RENDER_DPI
    resolution = RENDER_RESOLUTION

    s = [xyExtents[x_axis] * 100, xyExtents[y_axis] * 100]

    fig.set_size_inches((s[0] * resolution / 100 + 10) / fig.dpi, (s[1] * resolution / 100 + 10) / fig.dpi)
    ax2.set_xlim((0, s[0]))
    ax2.set_ylim((0, s[1]))
    ax2.scatter(points[:, x_axis] * 100, points[:, y_axis] * 100, c=colors, marker=',', lw=0, s=(sz / fig.dpi) ** 2)
    ax2.axis('off')
    fig.patch.set_facecolor('none')
    ax2.set_facecolor('none')
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, dpi=fig.dpi, transparent=True, format='png')
    buf.seek(0)

    image_data = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
    res = (img[5:-5, 5:-5])

    plt.close()
    plt.clf()
    plt.close(fig)

    return res


def read_ply_scalar(filepath):
    """Read the intensity scalar field from a PLY file, returns None if not present."""
    with open(filepath, 'rb') as f:
        # Parse header
        properties = []
        n_vertices = 0
        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                properties.append((parts[1], parts[2]))

        # Check if intensity exists
        has_intensity = any(name == 'intensity' for _, name in properties)
        if not has_intensity:
            return None

        # Calculate byte offsets
        type_sizes = {'float': 4, 'uchar': 1, 'double': 8, 'int': 4, 'short': 2}
        vertex_size = sum(type_sizes[t] for t, _ in properties)
        intensity_offset = 0
        for t, name in properties:
            if name == 'intensity':
                break
            intensity_offset += type_sizes[t]

        # Read all vertex data, extract intensity as strided float32 view
        data = np.frombuffer(f.read(n_vertices * vertex_size), dtype=np.uint8)
        scalars = np.ndarray(n_vertices, dtype=np.float32,
                             buffer=data, offset=intensity_offset,
                             strides=(vertex_size,))
        return scalars.copy()


def scalar_to_colors_slope(scalars):
    """Map slope scalar values to colors using config colormap params.
    Supports asymmetric ranges via SLOPE_RANGE_POSITIVE/NEGATIVE.
    Positive deviation (steeper) = more batter = blue (0).
    Negative deviation (shallower) = less batter = red (1)."""
    cmap = plt.cm.jet
    deviation = (scalars - EXPECTED_WALL_SLOPE) * 100  # percent
    range_pos = SLOPE_RANGE_POSITIVE if SLOPE_RANGE_POSITIVE is not None else SLOPE_COLORMAP_RANGE
    range_neg = SLOPE_RANGE_NEGATIVE if SLOPE_RANGE_NEGATIVE is not None else SLOPE_COLORMAP_RANGE
    # Map: +range_pos → 0 (blue), 0 → 0.5 (green), -range_neg → 1 (red)
    mapped = np.where(
        deviation >= 0,
        0.5 - 0.5 * np.clip(deviation / range_pos, 0, 1),
        0.5 + 0.5 * np.clip(-deviation / range_neg, 0, 1),
    )
    return cmap(mapped)[:, :3]


def scalar_to_colors_displacement(scalars):
    """Map displacement scalar values to colors using config colormap params.
    Supports asymmetric ranges via MAX_DISPLACEMENT_POSITIVE/NEGATIVE.
    Positive displacement = more batter (backward) = blue (0).
    Negative displacement = less batter (forward) = red (1)."""
    cmap = plt.cm.jet
    max_pos = MAX_DISPLACEMENT_POSITIVE if MAX_DISPLACEMENT_POSITIVE is not None else MAX_DISPLACEMENT_FOR_COLORS
    max_neg = MAX_DISPLACEMENT_NEGATIVE if MAX_DISPLACEMENT_NEGATIVE is not None else MAX_DISPLACEMENT_FOR_COLORS
    # Map: +max_pos → 0 (blue), 0 → 0.5 (green), -max_neg → 1 (red)
    mapped = np.where(
        scalars >= 0,
        0.5 - 0.5 * np.clip(scalars / max_pos, 0, 1),
        0.5 + 0.5 * np.clip(-scalars / max_neg, 0, 1),
    )
    return cmap(mapped)[:, :3]


def scalar_to_colors_deviation(scalars):
    """Map slope deviation scalar values to colors (already deviation from expected).
    Supports asymmetric ranges via SLOPE_RANGE_POSITIVE/NEGATIVE."""
    cmap = plt.cm.jet
    deviation = scalars * 100  # already deviation, convert to percent
    range_pos = SLOPE_RANGE_POSITIVE if SLOPE_RANGE_POSITIVE is not None else SLOPE_COLORMAP_RANGE
    range_neg = SLOPE_RANGE_NEGATIVE if SLOPE_RANGE_NEGATIVE is not None else SLOPE_COLORMAP_RANGE
    # Map: +range_pos → 0 (blue), 0 → 0.5 (green), -range_neg → 1 (red)
    mapped = np.where(
        deviation >= 0,
        0.5 - 0.5 * np.clip(deviation / range_pos, 0, 1),
        0.5 + 0.5 * np.clip(-deviation / range_neg, 0, 1),
    )
    return cmap(mapped)[:, :3]


def upscale_image_resize(image, scale_factor=10):
    """Upscale image using cv2.resize with bicubic interpolation."""
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upscaled


# ── Main execution ──────────────────────────────────────────────────────────

def render_point_cloud(points, colors, target, x_axis, y_axis, z_axis, sz, extents=None):
    """Render a set of points to a BGRA image."""
    if extents is None:
        extents = np.max(points, axis=0)
    return projectToImage1000_color(points, extents, colors, x_axis, y_axis, z_axis, sz=sz)


def station_align(points, x_axis):
    """Scale and offset x-coordinates so they map to station space (meters).

    After this transform, x=0 corresponds to station 0 and the data sits
    between STATION_START_OFFSET and (STATION_MAX - STATION_END_OFFSET).
    Returns (points_copy, total_extent_m) where total_extent_m is the full
    station range in meters.  If alignment is not configured, returns
    (points, None) unchanged.
    """
    if STATION_MAX_FT is None:
        return points, None
    total_m = STATION_MAX_FT * FEET_TO_METERS
    start_m = STATION_START_OFFSET_IN * 0.0254
    end_m = STATION_END_OFFSET_IN * 0.0254
    data_range_m = total_m - start_m - end_m

    x_max = np.max(points[:, x_axis]) - np.min(points[:, x_axis])
    if x_max <= 0:
        return points, None

    scale = data_range_m / x_max
    pts = points.copy()
    pts[:, x_axis] = (pts[:, x_axis] - np.min(pts[:, x_axis])) * scale + start_m
    return pts, total_m


def get_station_ranges(points, x_axis):
    """Return list of (start_m, end_m, label) tuples for station splits, or None for full image.

    Assumes points are already in station-aligned space (x=0 = station 0)
    if station alignment is configured.
    """
    if not STATION_SPLITS:
        return None
    ranges = []
    for start_ft, end_ft in STATION_SPLITS:
        start_m = start_ft * FEET_TO_METERS
        end_m = end_ft * FEET_TO_METERS
        label = "%d+%02d_to_%d+%02d" % (start_ft // 100, start_ft % 100, end_ft // 100, end_ft % 100)
        ranges.append((start_m, end_m, label))
    return ranges


# Map RENDER_TARGET keyword to glob pattern and file prefix
TARGET_GLOB = {
    "slope": "outputs/point_clouds/unrolled/slope_*.ply",
    "slope_threshold": "outputs/point_clouds/unrolled/slope_threshold_*.ply",
    "displacement": "outputs/point_clouds/unrolled/displacement_*.ply",
    "new_slope": "outputs/point_clouds/unrolled/new_slope_*.ply",
    "expected_slope": "outputs/point_clouds/unrolled/expected_slope_*.ply",
    "cross_section": "outputs/point_clouds/unrolled/cross_section_*.ply",
}

saveLoc = "outputs/images/"

for target in [RENDER_TARGET]:
    printf("Target: %s" % target)
    glob_pattern = TARGET_GLOB.get(target)
    if glob_pattern is None:
        printf("Unknown target: %s" % target)
        continue
    files = glob.glob(glob_pattern)
    for i in tqdm(range(len(files))):
        file = files[i]
        pc_source = o3d.t.io.read_point_cloud(file)
        pc_source = zero_pc(pc_source)
        points = pc_source.point.positions.numpy()

        # Recompute colors from scalar field if present
        scalars = read_ply_scalar(file)
        if scalars is not None and target in ("slope", "slope_threshold"):
            printf("Recomputing slope colors from scalar field")
            colors = scalar_to_colors_slope(scalars)
        elif scalars is not None and target == "new_slope":
            printf("Recomputing new_slope colors from scalar field")
            colors = scalar_to_colors_deviation(scalars)
        elif scalars is not None and target == "displacement":
            printf("Recomputing displacement colors from scalar field")
            colors = scalar_to_colors_displacement(scalars)
        else:
            colors = pc_source.point.colors.numpy()
            colors = colors.reshape(-1, 3)
            if colors.max() > 1.0:
                colors = colors / 255.0
        x_axis = 0
        y_axis = 2
        z_axis = 1
        if target == "cross_section":
            x_axis = 1
            y_axis = 2
            z_axis = 0
        if target == "displacement":
            sz = MARKER_SIZE_DISPLACEMENTS
        elif target in ("slope", "slope_threshold"):
            sz = MARKER_SIZE_SLOPES
        else:
            sz = MARKER_SIZE_DEFAULT

        # Extract name from filename (strip directory and extension)
        basename = os.path.splitext(os.path.basename(file))[0]

        if target == "cross_section":
            res = render_point_cloud(points, colors, target, x_axis, y_axis, z_axis, sz)
            res = cv2.flip(res, 1)
            ensure_dir('%s%s.png' % (saveLoc, basename))
            cv2.imwrite('%s%s.png' % (saveLoc, basename), res)
        else:
            # Align x-coordinates to station space
            points, total_m = station_align(points, x_axis)

            station_ranges = get_station_ranges(points, x_axis)
            if station_ranges:
                full_extents = np.max(points, axis=0)
                if total_m is not None:
                    full_extents[x_axis] = total_m
                for start_m, end_m, station_label in station_ranges:
                    mask = (points[:, x_axis] >= start_m) & (points[:, x_axis] < end_m)
                    sub_points = points[mask]
                    sub_colors = colors[mask]
                    if len(sub_points) == 0:
                        continue
                    # Shift x so this segment starts at 0
                    sub_points = sub_points.copy()
                    sub_points[:, x_axis] -= start_m
                    sub_extents = np.max(sub_points, axis=0)
                    sub_extents[x_axis] = end_m - start_m
                    sub_extents[y_axis] = full_extents[y_axis]
                    res = render_point_cloud(sub_points, sub_colors, target, x_axis, y_axis, z_axis, sz, extents=sub_extents)
                    ensure_dir('%s%s_%s.png' % (saveLoc, basename, station_label))
                    cv2.imwrite('%s%s_%s.png' % (saveLoc, basename, station_label), res)
            else:
                extents = np.max(points, axis=0)
                if total_m is not None:
                    extents[x_axis] = total_m
                res = render_point_cloud(points, colors, target, x_axis, y_axis, z_axis, sz, extents=extents)
                ensure_dir('%s%s.png' % (saveLoc, basename))
                cv2.imwrite('%s%s.png' % (saveLoc, basename), res)
