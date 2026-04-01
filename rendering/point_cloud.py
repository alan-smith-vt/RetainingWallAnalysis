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
)

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
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, dpi=fig.dpi)
    buf.seek(0)

    image_data = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    res = (img[5:-5, 5:-5])

    plt.close()
    plt.clf()
    plt.close(fig)

    return res


def upscale_image_resize(image, scale_factor=10):
    """Upscale image using cv2.resize with bicubic interpolation."""
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upscaled


# ── Main execution ──────────────────────────────────────────────────────────

for target in [RENDER_TARGET]:
    printf("Target: %s" % target)
    saveLoc = "renders/%s/" % target
    files = glob.glob("pointClouds/unrolled/%s/*.ply" % target)
    for i in tqdm(range(len(files))):
        file = files[i]
        pc_source = o3d.t.io.read_point_cloud(file)
        pc_source = zero_pc(pc_source)
        points = pc_source.point.positions.numpy()
        colors = pc_source.point.colors.numpy()
        colors = colors.reshape(-1, 3)
        colors[colors[:, 1] > 1] = np.array([1, 1, 1])
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
        res = projectToImage1000_color(points, np.max(points, axis=0), colors, x_axis, y_axis, z_axis, sz=sz)
        if target == "crossSections":
            name = file[-5:-4]
            res = cv2.flip(res, 1)
        else:
            pattern = r'unrolled_(\d+_-?\d+\.\d+(?:_\d+\.\d+)?)\.ply$'
            name = re.search(pattern, file).group(1)
        ensure_dir('%s%s.png' % (saveLoc, name))
        cv2.imwrite('%s%s.png' % (saveLoc, name), res)
