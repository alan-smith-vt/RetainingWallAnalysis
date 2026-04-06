"""
Cross section point cloud renderer.

Loads wall cross section, expected slope, and vertical reference PLYs,
recolors reference lines (magenta / black), combines them, and renders
station-split PNG images.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d
import numpy as np
import glob
from datetime import datetime
from tqdm import tqdm

from config import (
    RENDER_DPI, RENDER_RESOLUTION,
    MARKER_SIZE_DEFAULT,
    WALL_IDS, FEET_TO_METERS,
)
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

from rendering.point_cloud import (
    ensure_dir, printf, zero_pc,
    projectToImage1000_color, station_align, get_station_ranges,
    render_point_cloud,
)

# Cross section axes: Y along wall, Z elevation, X depth
X_AXIS = 1   # Y coordinate = along-wall
Y_AXIS = 2   # Z coordinate = elevation
Z_AXIS = 0   # X coordinate = depth (not rendered)

SAVE_LOC = "outputs/images/"


def load_ply(filepath):
    """Load a PLY and return (points, colors) with colors in 0-1 float range."""
    pc = o3d.t.io.read_point_cloud(filepath)
    points = pc.point.positions.numpy()
    colors = pc.point.colors.numpy().reshape(-1, 3)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return points, colors


def render_cross_section():
    import cv2

    ply_dir = "outputs/point_clouds/unrolled"

    for wall_id in WALL_IDS:
        wall_file = os.path.join(ply_dir, "cross_section_%d.ply" % wall_id)
        expected_file = os.path.join(ply_dir, "cross_section_expected_%d.ply" % wall_id)
        vertical_file = os.path.join(ply_dir, "cross_section_vertical_%d.ply" % wall_id)

        if not os.path.exists(wall_file):
            printf("Cross section PLY not found: %s" % wall_file)
            continue

        printf("Loading cross section PLYs for wall %d" % wall_id)

        # Load wall points (keep displacement colors)
        wall_pts, wall_colors = load_ply(wall_file)

        # Load expected slope, recolor to magenta (1, 0, 1)
        if os.path.exists(expected_file):
            exp_pts, _ = load_ply(expected_file)
            exp_colors = np.full((len(exp_pts), 3), [1.0, 0.0, 1.0])
        else:
            exp_pts = np.zeros((0, 3))
            exp_colors = np.zeros((0, 3))
            printf("  Expected slope PLY not found, skipping")

        # Load vertical reference, recolor to black (0, 0, 0)
        if os.path.exists(vertical_file):
            vert_pts, _ = load_ply(vertical_file)
            vert_colors = np.full((len(vert_pts), 3), [0.0, 0.0, 0.0])
        else:
            vert_pts = np.zeros((0, 3))
            vert_colors = np.zeros((0, 3))
            printf("  Vertical reference PLY not found, skipping")

        # Combine all points
        all_pts = np.vstack([wall_pts, exp_pts, vert_pts])
        all_colors = np.vstack([wall_colors, exp_colors, vert_colors])

        # Zero all points together
        min_coords = np.min(all_pts, axis=0)
        all_pts = all_pts - min_coords

        printf("  Combined %d points (%d wall, %d expected, %d vertical)" %
               (len(all_pts), len(wall_pts), len(exp_pts), len(vert_pts)))

        # Station align
        all_pts, total_m = station_align(all_pts, X_AXIS)

        station_ranges = get_station_ranges(all_pts, X_AXIS)
        basename = "cross_section_%d" % wall_id

        sz = MARKER_SIZE_DEFAULT

        if station_ranges:
            full_extents = np.max(all_pts, axis=0)
            if total_m is not None:
                full_extents[X_AXIS] = total_m

            for start_m, end_m, station_label in tqdm(station_ranges, desc="Wall %d stations" % wall_id):
                mask = (all_pts[:, X_AXIS] >= start_m) & (all_pts[:, X_AXIS] < end_m)
                sub_pts = all_pts[mask].copy()
                sub_colors = all_colors[mask]
                if len(sub_pts) == 0:
                    continue

                sub_pts[:, X_AXIS] -= start_m
                sub_extents = np.max(sub_pts, axis=0)
                sub_extents[X_AXIS] = end_m - start_m
                sub_extents[Y_AXIS] = full_extents[Y_AXIS]

                res = render_point_cloud(sub_pts, sub_colors, "cross_section",
                                         X_AXIS, Y_AXIS, Z_AXIS, sz, extents=sub_extents)
                res = cv2.flip(res, 1)
                out_path = "%s%s_%s.png" % (SAVE_LOC, basename, station_label)
                ensure_dir(out_path)
                cv2.imwrite(out_path, res)
        else:
            extents = np.max(all_pts, axis=0)
            if total_m is not None:
                extents[X_AXIS] = total_m
            res = render_point_cloud(all_pts, all_colors, "cross_section",
                                     X_AXIS, Y_AXIS, Z_AXIS, sz, extents=extents)
            res = cv2.flip(res, 1)
            out_path = "%s%s.png" % (SAVE_LOC, basename)
            ensure_dir(out_path)
            cv2.imwrite(out_path, res)

        printf("  Done rendering wall %d" % wall_id)


if __name__ == "__main__":
    render_cross_section()
