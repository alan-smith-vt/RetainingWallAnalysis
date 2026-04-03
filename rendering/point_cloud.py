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
import struct
from tqdm import tqdm
from config import (
    RENDER_DPI, RENDER_RESOLUTION, RENDER_TARGET,
    MARKER_SIZE_DEFAULT, MARKER_SIZE_DISPLACEMENTS, MARKER_SIZE_SLOPES,
    EXPECTED_WALL_SLOPE, SLOPE_COLORMAP_RANGE, MAX_DISPLACEMENT_FOR_COLORS,
    FEET_TO_METERS,
)
try:
    from config import STATION_SPLITS
except ImportError:
    STATION_SPLITS = None

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
        type_formats = {'float': 'f', 'uchar': 'B', 'double': 'd', 'int': 'i', 'short': 'h'}
        vertex_size = sum(type_sizes[t] for t, _ in properties)
        intensity_offset = 0
        for t, name in properties:
            if name == 'intensity':
                break
            intensity_offset += type_sizes[t]

        # Read all vertices and extract intensity
        data = f.read(n_vertices * vertex_size)
        scalars = np.zeros(n_vertices, dtype=np.float32)
        for i in range(n_vertices):
            offset = i * vertex_size + intensity_offset
            scalars[i] = struct.unpack_from('f', data, offset)[0]
        return scalars


def scalar_to_colors_slope(scalars):
    """Map slope scalar values to colors using config colormap params."""
    cmap = plt.cm.jet
    deviation = scalars - EXPECTED_WALL_SLOPE
    mapped = np.clip((-deviation * 100) / SLOPE_COLORMAP_RANGE * 0.5 + 0.5, 0, 1)
    return cmap(mapped)[:, :3]


def scalar_to_colors_displacement(scalars):
    """Map displacement scalar values to colors using config colormap params."""
    cmap = plt.cm.jet
    mapped = np.clip(-scalars / MAX_DISPLACEMENT_FOR_COLORS * 0.5 + 0.5, 0, 1)
    return cmap(mapped)[:, :3]


def scalar_to_colors_deviation(scalars):
    """Map slope deviation scalar values to colors (already deviation from expected)."""
    cmap = plt.cm.jet
    mapped = np.clip((-scalars * 100) / SLOPE_COLORMAP_RANGE * 0.5 + 0.5, 0, 1)
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


def get_station_ranges(points, x_axis):
    """Return list of (start_m, end_m, label) tuples for station splits, or None for full image."""
    if not STATION_SPLITS:
        return None
    x_min = np.min(points[:, x_axis])
    ranges = []
    for start_ft, end_ft in STATION_SPLITS:
        start_m = start_ft * FEET_TO_METERS + x_min
        end_m = end_ft * FEET_TO_METERS + x_min
        label = "%d+%02d_to_%d+%02d" % (start_ft // 100, start_ft % 100, end_ft // 100, end_ft % 100)
        ranges.append((start_m, end_m, label))
    return ranges


for target in [RENDER_TARGET]:
    printf("Target: %s" % target)
    saveLoc = "renders/%s/" % target
    files = glob.glob("pointClouds/unrolled/%s/*.ply" % target)
    for i in tqdm(range(len(files))):
        file = files[i]
        pc_source = o3d.t.io.read_point_cloud(file)
        pc_source = zero_pc(pc_source)
        points = pc_source.point.positions.numpy()

        # Recompute colors from scalar field if present
        scalars = read_ply_scalar(file)
        if scalars is not None and target in ("slopes", "slopes/thresholds/"):
            printf("Recomputing slope colors from scalar field")
            colors = scalar_to_colors_slope(scalars)
        elif scalars is not None and target == "new_slopes":
            printf("Recomputing new_slope colors from scalar field")
            colors = scalar_to_colors_deviation(scalars)
        elif scalars is not None and target == "displacements":
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
        if target == "crossSections":
            x_axis = 1
            y_axis = 2
            z_axis = 0
        if target == "displacements":
            sz = MARKER_SIZE_DISPLACEMENTS
        elif target == "slopes" or "slopes/thresholds/":
            sz = MARKER_SIZE_SLOPES
        else:
            sz = MARKER_SIZE_DEFAULT

        if target == "crossSections":
            name = file[-5:-4]
            res = render_point_cloud(points, colors, target, x_axis, y_axis, z_axis, sz)
            res = cv2.flip(res, 1)
            ensure_dir('%s%s.png' % (saveLoc, name))
            cv2.imwrite('%s%s.png' % (saveLoc, name), res)
        else:
            pattern = r'unrolled_(\d+_-?\d+\.\d+(?:_\d+\.\d+)?)\.ply$'
            name = re.search(pattern, file).group(1)
            station_ranges = get_station_ranges(points, x_axis)
            if station_ranges:
                full_extents = np.max(points, axis=0)
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
                    sub_extents[y_axis] = full_extents[y_axis]
                    res = render_point_cloud(sub_points, sub_colors, target, x_axis, y_axis, z_axis, sz, extents=sub_extents)
                    ensure_dir('%s%s_%s.png' % (saveLoc, name, station_label))
                    cv2.imwrite('%s%s_%s.png' % (saveLoc, name, station_label), res)
            else:
                res = render_point_cloud(points, colors, target, x_axis, y_axis, z_axis, sz)
                ensure_dir('%s%s.png' % (saveLoc, name))
                cv2.imwrite('%s%s.png' % (saveLoc, name), res)
