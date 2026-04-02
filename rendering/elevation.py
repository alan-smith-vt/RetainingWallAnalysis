"""
Elevation profile renderer.

Extracts a horizontal slice of the displacement point cloud at the reference
elevation (+-1 foot), applies vertical exaggeration, and renders it to a PNG.
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
    RENDER_DPI, RENDER_RESOLUTION, REFERENCE_ELEVATION_M,
    ELEVATION_WINDOW_M, VERTICAL_EXAGGERATION, ELEVATION_MARKER_SIZE,
    FEET_TO_METERS,
)
try:
    from config import STATION_SPLITS
except ImportError:
    STATION_SPLITS = None

print("[%s]: Python started." % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


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


def savePoints(points, filePath, colors=None):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = points
    if colors is not None:
        pcd.point.colors = colors
    ensure_dir(filePath)
    o3d.t.io.write_point_cloud(filePath, pcd)


# ── Main execution ──────────────────────────────────────────────────────────

target = "displacements"
printf("Target: %s" % target)
saveLoc = "renders/elevations/"
files = glob.glob("pointClouds/unrolled/%s/*0.1.ply" % target)
for i in tqdm(range(len(files))):
    file = files[i]
    pc_source = o3d.t.io.read_point_cloud(file)
    points = pc_source.point.positions.numpy()
    colors = pc_source.point.colors.numpy()
    points_og = points.copy()
    points_og_zeroed = points_og - np.min(points_og, axis=0)

    ref_z = REFERENCE_ELEVATION_M
    delta_z = ELEVATION_WINDOW_M
    inds = (points[:, 2] < ref_z + delta_z) & (points[:, 2] > ref_z - delta_z)
    points = points[inds]

    og_offset = np.min(points_og, axis=0)
    og_offset[2] = ref_z - delta_z
    points = points - og_offset

    scale = VERTICAL_EXAGGERATION
    points[:, 2] = points[:, 2] * scale
    colors = colors[inds]

    colors = colors.reshape(-1, 3)
    colors[colors[:, 1] > 1] = np.array([1, 1, 1])

    x_axis = 0
    y_axis = 2
    z_axis = 1
    sz = ELEVATION_MARKER_SIZE

    pattern = r'unrolled_(\d+_-?\d+\.\d+(?:_\d+\.\d+)?)\.ply$'
    name = re.search(pattern, file).group(1)
    points_save = points.copy()
    points_save[:, 1] = points_save[:, 1] * 0
    savePoints(points_save, "exagerated_%s.ply" % name)

    extents = np.max(points_og_zeroed, axis=0)
    extents[2] = delta_z * 2 * scale

    if STATION_SPLITS:
        x_min_og = np.min(points_og[:, x_axis])
        for start_ft, end_ft in STATION_SPLITS:
            start_m = start_ft * FEET_TO_METERS + x_min_og - og_offset[x_axis]
            end_m = end_ft * FEET_TO_METERS + x_min_og - og_offset[x_axis]
            mask = (points[:, x_axis] >= start_m) & (points[:, x_axis] < end_m)
            sub_points = points[mask]
            sub_colors = colors[mask]
            if len(sub_points) == 0:
                continue
            sub_points = sub_points.copy()
            sub_points[:, x_axis] -= start_m
            sub_extents = extents.copy()
            sub_extents[x_axis] = end_m - start_m
            station_label = "%d+%02d_to_%d+%02d" % (start_ft // 100, start_ft % 100, end_ft // 100, end_ft % 100)
            res = projectToImage1000_color(sub_points, sub_extents, sub_colors, x_axis, y_axis, z_axis, sz=sz)
            ensure_dir('%s%s_%s_elevation.png' % (saveLoc, name, station_label))
            cv2.imwrite('%s%s_%s_elevation.png' % (saveLoc, name, station_label), res)
    else:
        res = projectToImage1000_color(points, extents, colors, x_axis, y_axis, z_axis, sz=sz)
        ensure_dir('%s%s_elevation.png' % (saveLoc, name))
        cv2.imwrite('%s%s_elevation.png' % (saveLoc, name), res)
