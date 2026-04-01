"""
Point cloud downsampler.

Reduces point cloud density using voxel downsampling for faster
downstream processing.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d
from datetime import datetime
from config import DOWNSAMPLE_VOXEL_SIZE, DOWNSAMPLE_INPUT, DOWNSAMPLE_OUTPUT

print("[%s]: Python started." % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


pc_source = o3d.t.io.read_point_cloud(DOWNSAMPLE_INPUT)
printf("Loaded pc")

pc_source = pc_source.voxel_down_sample(voxel_size=DOWNSAMPLE_VOXEL_SIZE)
printf("Downsampled pc")
o3d.t.io.write_point_cloud(DOWNSAMPLE_OUTPUT, pc_source)
printf("Saved downsampled pc")
