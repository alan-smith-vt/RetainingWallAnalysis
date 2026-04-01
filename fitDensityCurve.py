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

def savePoints(points, filePath, color=None):
	pcd = o3d.t.geometry.PointCloud()
	pcd.point.positions = points
	if color is not None:
		colors = np.tile(color, (len(points), 1))
		pcd.point.colors = colors
	o3d.t.io.write_point_cloud(filePath, pcd)

#Fails with sloped roofs. Need to rotate based on the unfiltered region surface so density axis is correct.
def extractTile_maxDensity(npPoints, bin_width, slice_axis, debug=False):
	sorted_indices = np.argsort(npPoints[:,slice_axis])
	sorted_points_raw = npPoints[sorted_indices]
	#Normalize to 0 minimum
	sorted_points = sorted_points_raw - sorted_points_raw[0,slice_axis]
	#printf("Sorted points")

	z_min = sorted_points[0,slice_axis]
	z_max = sorted_points[-1,slice_axis]
	z_range = z_max-z_min
	#bin_width = 0.01#1 cm
	num_slices = int(z_range//bin_width)
	z_values = sorted_points[:,slice_axis]

	#Count points between each "bin" range
	z_density = []
	prev_idx = 0
	for i in range(1,num_slices):
		# Use binary search to find the insertion index
		target_z = bin_width*i
		next_idx = np.searchsorted(z_values, target_z)
		z_density.append(next_idx-prev_idx)
		prev_idx = next_idx

	z_density_array = np.array(z_density)
	x_axis = np.linspace(0,1,num_slices)*z_range-z_min
	x_axis_density = x_axis[1:]-bin_width/2

	#Find the indices at 1 cm above and below the max density
	if len(z_density_array)==0:
		return None
	max_density = np.max(z_density_array)
	max_density_idx = z_density.index(max_density)
	

	loc_cm = x_axis_density[max_density_idx]
	crop_delta = 0.01#  ±1 cm
	cm_above = loc_cm+crop_delta
	cm_below = loc_cm-crop_delta

	idx_above = np.searchsorted(x_axis_density, cm_above)
	idx_below = np.searchsorted(x_axis_density, cm_below)

	if idx_above >= len(x_axis_density):
		idx_above = len(x_axis_density) - 1

	#tile = sorted_points_raw[idx_below:idx_above]

	if debug:
		printf("Counted points in each bin")
		#Plot "density curve"
		plt.plot(x_axis_density,z_density_array)
		plt.scatter(loc_cm, max_density)
		#plt.scatter(x_axis_density[[idx_below, idx_above]],z_density_array[[idx_below, idx_above]])
		plt.legend(["Density","Max", "Chosen Splits"])
		plt.show()

	#Recreate the sorted points (since we don't want them zeroed for this)
	sorted_points = npPoints[sorted_indices]
	z_min = sorted_points[0,slice_axis]
	loc_cm = loc_cm+z_min

	return loc_cm

def sortAndSlice_old(npPoints, bin_width, axis):
	sorted_indices = np.argsort(npPoints[:,axis])
	sorted_points = npPoints[sorted_indices]
	#Normalize to 0 minimum
	#sorted_points = sorted_points - sorted_points[0,axis]
	#printf("Sorted points for axis %d"%axis)

	z_min = sorted_points[0,axis]
	z_max = sorted_points[-1,axis]
	z_range = z_max-z_min
	num_slices = int(z_range//bin_width)
	z_values = sorted_points[:,axis]

	#Count points between each "bin" range
	slice_points = []
	prev_idx = 0
	i = 1
	target_z = z_min + bin_width*i
	#print("Target z: %3.3f, z_max: %3.3f"%(target_z,z_max))
	while True: 
		#for i in range(1,num_slices):
		# Use binary search to find the insertion index
		target_z = z_min + bin_width*i
		next_idx = np.searchsorted(z_values, target_z)
		slice_points.append(sorted_points[prev_idx:next_idx])
		prev_idx = next_idx
		i = i + 1
		if target_z > z_max:
			break

	#Don't forget the last slice
	#slice_points.append(sorted_points[prev_idx:])

	return slice_points

def sortAndSlice(npPoints, window_thickness, step_size, axis):
    sorted_indices = np.argsort(npPoints[:,axis])
    sorted_points = npPoints[sorted_indices]
    
    z_min = sorted_points[0,axis]
    z_max = sorted_points[-1,axis]
    z_range = z_max - z_min
    z_values = sorted_points[:,axis]
    
    slice_points = []
    
    # Start the first window at z_min
    window_start = z_min
    
    while window_start <= z_max:
        window_end = window_start + window_thickness
        
        # Find indices for points within this window [window_start, window_end)
        start_idx = np.searchsorted(z_values, window_start, side='left')
        end_idx = np.searchsorted(z_values, window_end, side='left')
        
        # Only add slice if it contains points
        if start_idx < end_idx:
            slice_points.append(sorted_points[start_idx:end_idx])
        
        # Move to next window position
        window_start += step_size
        
        # Break if we've moved past the data range
        if window_start > z_max:
            break
    
    return slice_points


files = glob.glob("pointClouds/unrolled/elevations/cuts/*0.1.ply")
#file = "pointClouds/unrolled/elevations/cuts/exagerated_2_0.1.ply"
for file in files:
	wall_id = file[-9]
	pc_source = o3d.t.io.read_point_cloud(file)
	points = pc_source.point.positions.numpy()
	x_min = np.min(points,axis=0)[0]
	window = 0.5
	step = 1
	slices = sortAndSlice(points, window, step, 0)
	#slice_test = slices[20].copy()
	#savePoints(slice_test,"test.ply")
	peaks = []
	debug=False
	for i in tqdm(range(len(slices)-1)):
		peak = extractTile_maxDensity(slices[i], 0.2, 2, debug=debug)
		if not peak:
			continue
		peaks.append(np.array([x_min+step*i,0,peak]))
	peaks.append(np.array([x_min+step*len(slices)-1,0,peak]))
	peaks = np.vstack(peaks)
	savePoints(peaks,"pointClouds/unrolled/elevations/curves/%s_peaks.ply"%wall_id,color=np.array([255,0,255],dtype=np.uint8))

	from scipy.interpolate import splprep, splev
	x = peaks[:,0]
	y = peaks[:,2]
	# Parametric smoothing spline
	smoothing_factor = 5
	tck, u = splprep([x, y], s=smoothing_factor)  # s>0 for smoothing

	#second pass:
	u_vals = x/np.max(x)
	_, y_smooth = splev(u_vals, tck)
	residuals = np.abs(y-y_smooth)
	inds = residuals>0.5#threshold
	x = x[~inds]
	y = y[~inds]

	smoothing_factor = 1
	tck, u = splprep([x, y], s=smoothing_factor)  # s>0 for smoothing


	u_new = np.linspace(0,1,1000)
	x_smooth, y_smooth = splev(u_new, tck)
	peaks_smooth = np.zeros((len(x_smooth),3))
	peaks_smooth[:,0] = x_smooth
	peaks_smooth[:,2] = y_smooth

	savePoints(peaks_smooth,"pointClouds/unrolled/elevations/curves/%s.ply"%wall_id, color=np.array([0,0,255],dtype=np.uint8))

	x_sample = np.arange(0,np.max(x),1)
	x_sample = x_sample[x_sample>np.min(x)]
	x_sample = np.insert(x_sample, 0, np.min(x))
	x_sample = np.append(x_sample, np.max(x))
	# x_sample = np.linspace(np.min(x),np.max(x),100)

	u_sample = x_sample/np.max(x)
	_,y_sample = splev(u_sample, tck)

	delta_z = 0.3048#1 foot
	scale = 20
	y_sample = (y_sample/(scale*delta_z)+760.5)

	res = np.column_stack([x_sample*3.28, y_sample])#convert meters to feet


	np.savetxt("excel_data/wall_%s_settlement.csv"%wall_id,res,delimiter=",")
