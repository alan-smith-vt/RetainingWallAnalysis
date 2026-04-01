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

print("[%s]: Python started."%datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def printf(msg):
	print("[%s]: %s"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))

def zero_pc(pc, ref_pc=None):
	points = pc.point.positions.numpy()
	if ref_pc==None:
		points = points-np.min(points,axis=0)
	else:
		ref_points = ref_pc.point.positions.numpy()
		points = points-np.min(ref_points,axis=0)
	colors = pc.point.colors.numpy()

	pc_zeroed = o3d.t.geometry.PointCloud()
	pc_zeroed.point.positions = points
	pc_zeroed.point.colors = colors
	return pc_zeroed

#Convert a set of points from 3d physical space to a 2d image representation
def projectToImage1000_color(points, xyExtents, colors, x_axis, y_axis, z_axis,sz=72.):
	#x_min = np.min(grid_points[:,0])
	#y_min = np.min(grid_points[:,1])

	#points[:,:2] = points[:,:2] - np.array([x_min, y_min])
	fig = plt.figure()
	ax2 = plt.gca()
	fig.dpi=10
	resolution = 100

	#print("color shape: %d, %d, data type: %s"%(colors.shape[0], colors.shape[1], type(colors[0,0])))

	#axis limits, originally set using the bounds of the total point cloud
	#This should probably be fixed at 1x1 meter * 100
	#s = [np.max(points[:,x_ax])*100,np.max(points[:,y_ax])*100]
	s = [xyExtents[x_axis]*100,xyExtents[y_axis]*100]

	fig.set_size_inches((s[0]*resolution/100+10)/fig.dpi,(s[1]*resolution/100+10)/fig.dpi)
	#fig.patch.set_facecolor('gray')  # Set figure background to black
	ax2.set_xlim((0,s[0]))
	ax2.set_ylim((0,s[1]))
	#printf("Making figure")
	ax2.scatter(points[:,x_axis]*100,points[:,y_axis]*100, c=colors, marker=',', lw=0, s=(sz/fig.dpi)**2)#72 default for s/fig.dpi
	#ax2.set_facecolor('gray')
	#printf("Figure to image")
	ax2.axis('off')
	fig.tight_layout()
	#printf("Saving figure")


	buf = BytesIO()
	fig.savefig(buf, dpi=fig.dpi)
	#fig.savefig("test_img.png", dpi=fig.dpi)

	buf.seek(0)
	#printf("Loading figure")

	image_data = np.frombuffer(buf.read(), dtype=np.uint8)
	img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
	#img = cv2.imread("test_img.tiff")
	res = (img[5:-5,5:-5])
	#res = 255-res#invert colors

	plt.close()
	plt.clf()
	plt.close(fig)
	
	return res

# Method 1: Using cv2.resize() with interpolation
def upscale_image_resize(image, scale_factor=10):
	"""
	Upscale image using cv2.resize with different interpolation methods
	"""
	height, width = image.shape[:2]
	new_width = int(width * scale_factor)
	new_height = int(height * scale_factor)

	# Different interpolation methods:
	# cv2.INTER_LINEAR - bilinear interpolation (default, good for most cases)
	# cv2.INTER_CUBIC - bicubic interpolation (smoother, slower)
	# cv2.INTER_LANCZOS4 - Lanczos interpolation (high quality)
	# cv2.INTER_NEAREST - nearest neighbor (pixelated, fastest)

	upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return upscaled

def savePoints(points, filePath, colors=None):
	pcd = o3d.t.geometry.PointCloud()
	pcd.point.positions = points
	if colors is not None:
		pcd.point.colors = colors
	o3d.t.io.write_point_cloud(filePath, pcd)

target = "displacements"
printf("Target: %s"%target)
saveLoc = "renders/elevations/"
files = glob.glob("pointClouds/unrolled/%s/*0.1.ply"%target)
for i in tqdm(range(len(files))):
	file = files[i]
	pc_source = o3d.t.io.read_point_cloud(file)
	points = pc_source.point.positions.numpy()
	colors = pc_source.point.colors.numpy()
	points_og = points.copy()
	points_og_zeroed = points_og - np.min(points_og,axis=0)

	ref_z = 232.1052#761.5 feet
	delta_z = 0.3048#1 foot
	inds = (points[:,2]<ref_z+delta_z)&(points[:,2]>ref_z-delta_z)
	points = points[inds]
	

	#zero the point cloud slice to the original point cloud's 
	#extents but fix the minimum z to our reference elevation
	og_offset = np.min(points_og,axis=0)
	og_offset[2] = ref_z-delta_z
	points = points-og_offset
	#break

	scale = 20
	points[:,2] = points[:,2]*scale
	colors = colors[inds]

	colors = colors.reshape(-1, 3)
	colors[colors[:,1]>1]=np.array([1,1,1])

	x_axis = 0
	y_axis = 2
	z_axis = 1
	sz=500

	pattern = r'unrolled_(\d+_-?\d+\.\d+(?:_\d+\.\d+)?)\.ply$'
	name = re.search(pattern, file).group(1)
	points_save = points.copy()
	points_save[:,1] = points_save[:,1]*0
	savePoints(points_save,"exagerated_%s.ply"%name)

	extents = np.max(points_og_zeroed,axis=0)
	extents[2] = delta_z*2*scale

	res = projectToImage1000_color(points, extents, colors, x_axis, y_axis, z_axis, sz=sz)
	# Upscale the image by factor of 10
	#res_upscaled = upscale_image_resize(res, scale_factor=5)

	
	cv2.imwrite('%s%s_elevation.png'%(saveLoc,name),res)
	#break
	#printf("Image written")

