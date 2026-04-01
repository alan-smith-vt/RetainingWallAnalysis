import open3d as o3d
from datetime import datetime
print("[%s]: Python started."%datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def printf(msg):
	print("[%s]: %s"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))

pc_source = o3d.t.io.read_point_cloud("KCLN Point Cloud - GZA.ply")
printf("Loaded pc")

pc_source = pc_source.voxel_down_sample(voxel_size=0.04)
printf("Downsampled pc")
o3d.t.io.write_point_cloud("PC_down_04.ply", pc_source)
printf("Saved downsampled pc")