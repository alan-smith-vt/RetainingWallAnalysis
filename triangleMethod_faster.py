import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import math
print("[%s]: Python started."%datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def printf(msg):
	print("[%s]: %s"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def savePoints(points, filePath, colors=None):
	pcd = o3d.t.geometry.PointCloud()
	pcd.point.positions = points
	if colors is not None:
		pcd.point.colors = colors
	o3d.t.io.write_point_cloud(filePath, pcd)

def getCrossSection(slicePoints, linePoints, filePath, k, slopeColor):
	#pcd = o3d.t.geometry.PointCloud()
	points = np.vstack([slicePoints, linePoints])
	points[:,1] = points[:,1] - np.min(points[:,1]) - k
	#pcd.point.positions = points
	colors_1 = np.ones_like(slicePoints)*np.array([255,255,255])
	colors_2 = np.ones_like(linePoints)*np.array(slopeColor)
	colors = np.vstack([colors_1,colors_2])
	#pcd.point.colors = colors
	#o3d.t.io.write_point_cloud(filePath, pcd)
	return points, colors

def fixSpacing(points, spacing=1):
	points = np.vstack([points,points[-1,:]])
	# Calculate cumulative distances along the line
	distances = np.zeros(len(points))
	for i in range(1, len(points)):
		distances[i] = distances[i-1] + np.linalg.norm(points[i] - points[i-1])
	# Create new equally spaced distances
	total_length = distances[-1]
	num_points = int(total_length / spacing) + 1
	new_distances = np.linspace(0, total_length, num_points)
	# Interpolate to get new vertices
	new_vertices = np.zeros((num_points, 3))
	for i in range(3):  # x, y, z coordinates
		new_vertices[:, i] = np.interp(new_distances, distances, points[:, i])
	return new_vertices

def hex_to_rgb(hex_color):
	"""Convert hex color to RGB array."""
	hex_color = hex_color.lstrip('#')
	return np.array([int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)])

def value_to_rgb_discrete(percent):
	"""Map a float value to unique hex colors based on percentage ranges."""
	percent = -percent#bandaid because colors backwards
	ranges = [
		(-float('inf'), -0.01, '#df03fc'), #bright purple
		(-0.01, 0.01, '#03fc24'), #bright green
		(0.01, 0.02, '#03fcf4'), #light blue
		(0.02, 0.03, '#2803fc'), #dark blue
		(0.03, 0.04, '#fccf03'), #light orange
		(0.04, 0.05, '#fc5e03'), #dark orange
		(0.05, float('inf'), '#fc0303') #bright red
	]
	
	for low, high, color in ranges:
		if low <= percent < high or (high == 0.01 and percent <= high):
			return hex_to_rgb(color)


def value_to_rgb_jet(percent):
	if percent > 0:
		return hex_to_rgb('#000000')#black
	cmap = plt.cm.jet
	percent = (-percent*100)/3.5#map 0 to -5% as 0 to 1 for cmap
	return cmap(percent)[:3]

def value_to_rgb(percent, thresh):
	if thresh is None:
		return value_to_rgb_jet(percent)
	#thresh = -0.015
	if percent < thresh:
		percent = thresh
		cmap = plt.cm.jet
		percent = (-percent*100)/3.5
		return cmap(percent)[:3]
	else:
		return hex_to_rgb('#ffffff')

def fitLineZ1toZ2(points_2d, z1, z2, x1, x2, thresh):
		z_values = points_2d[:,1]

		j1= np.searchsorted(z_values, z1)
		j2 = np.searchsorted(z_values, z2)

		sub_points_2d = points_2d[j1:j2,:]
		if len(sub_points_2d) < 10:
			return None, None, None, None, None

		# Fit line using least squares
		# Line equation: Y = mZ + b
		y_coords = sub_points_2d[:, 0]
		z_coords = sub_points_2d[:, 1]

		# Solve for slope and intercept
		A = np.vstack([z_coords, np.ones(len(z_coords))]).T
		slope, intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]

		#printf(f"Line equation: Y = {slope:.4f}*Z + {intercept:.4f}")

		#Line in 3d:
		z = np.linspace(z1, z2, 100)
		y = slope*z+intercept

		# Create 10 parallel lines from x1 to x2

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

		# If you need them as single arrays, you can concatenate:
		all_lines = np.vstack(lines)
		all_line_colors = np.vstack(line_colors_list)

		return slope, intercept, y, all_lines, all_line_colors

def unrollSlices(pc_slices, spacing):
	#for each point cloud slice
	for i in range(len(pc_slices)):
		axis = 0
		#zero the slice along the x axis then shift by number of spaces
		x_min = np.min(pc_slices[i][:,axis])
		pc_slices[i][:,axis] = (pc_slices[i][:,axis]-x_min)+spacing*i
	return np.vstack(pc_slices)

points_file = "pointClouds/KCLN_noColorCrop2_4cm.ply"

#
thresh = None#percent slope
#wall_id = 12#wall 1 end 2
#spacing = 1#meters per slice
#for thresh in [-0.01, -0.015, -0.02, -0.025, -0.03, -0.035]:
for spacing in [1]:
	print("~"*50)
	printf("Processing retaining walls with %.1f slice spacing"%spacing)
	print("~"*50)
	for wall_id in [1]:
		print("~"*50)
		printf("Starting Wall %d"%wall_id)
		print("~"*50)
		if wall_id == 1:
			vertices_file = "pointClouds/wall_1_vertices_all.ply"
		elif wall_id == 2:
			vertices_file = "pointClouds/wall_2_vertices_all.ply"
		elif wall_id == 3:
			vertices_file = "pointClouds/wall_3_vertices_all.ply"
		#vertices_file = "pointClouds/Vertices_centerWall.ply"

		pc_source = o3d.t.io.read_point_cloud(points_file)
		points = pc_source.point.positions.numpy()

		vertices_source = o3d.t.io.read_point_cloud(vertices_file)
		vertices = vertices_source.point.positions.numpy()
		vertices = fixSpacing(vertices, spacing=spacing)

		lines = []
		line_colors = []
		pc_slices = []
		pc_slice_colors = []
		slope_values = []
		pc_slope_colors = []
		pc_slices_rotated = []
		lines_rotated = []
		new_slopes = []
		new_slope_colors = []

		segLength = 10#meters

		#i = 0
		points_source = points.copy()
		
		# Pre-filter points once to avoid repeated filtering
		global_z_min = np.min(vertices[:,2]) - 1.0  # Buffer for safety
		points_master = points_source[points_source[:,2] > global_z_min,:]
		
		for i in tqdm(range(len(vertices)-1)):
			v1 = vertices[i]
			v2 = vertices[i+1]

			z_min = np.min([vertices[i,2], vertices[i+1,2]])#hardcoded ground plane
			# Use boolean masking instead of array filtering
			z_mask = points_master[:,2] > z_min
			points = points_master[z_mask,:]

			# Calculate rotation angle to align v1->v2 with x-axis
			direction = v2 - v1
			angle = -np.arctan2(direction[1], direction[0])

			# Create rotation matrix about z-axis
			cos_a, sin_a = np.cos(angle), np.sin(angle)
			rotation_matrix = np.array([[cos_a, -sin_a, 0],
									   [sin_a,  cos_a, 0],
									   [0,      0,     1]])

			# To undo the rotation:
			cos_a_inv, sin_a_inv = np.cos(-angle), np.sin(-angle)
			inverse_rotation_matrix = np.array([[cos_a_inv, -sin_a_inv, 0],
											   [sin_a_inv,  cos_a_inv, 0],
											   [0,          0,          1]])

			# Apply rotation to points
			points_rotated = np.dot(points - v1, rotation_matrix.T) + v1
			vertices_rotated = np.dot(vertices - v1, rotation_matrix.T) + v1

			v1_rotated = vertices_rotated[i]
			v2_rotated = vertices_rotated[i+1]

			#Undo operation for reference:
			#points_original = np.dot(points_rotated - v1, inverse_rotation_matrix.T) + v1

			#Extract points between v1x and v2x - use boolean masking instead of sorting
			x_values = points_rotated[:,0]  # Don't sort yet
			x_min, x_max = min(v1_rotated[0], v2_rotated[0]), max(v1_rotated[0], v2_rotated[0])
			
			# Use boolean masking instead of searchsorted + slicing
			x_mask = (x_values >= x_min) & (x_values <= x_max)
			pc_slice = points_rotated[x_mask]

			#Extract points within 2 meters of v1y
			y_mask = np.abs(pc_slice[:, 1] - v1_rotated[1]) <= 0.5
			pc_slice = pc_slice[y_mask]

			# Drop x dimension to get YZ plane points
			points_2d = pc_slice[:, 1:3]  # columns 1 and 2 (Y and Z)
			sorted_indices = np.argsort(points_2d[:,1]) # sort by z
			points_2d = points_2d[sorted_indices]

			z_max = np.max(pc_slice[:,2])-0.25#offset to remove the lip at the top of the wall
			x_val = (v1_rotated[0]+v2_rotated[0])/2

			numSlopes = max(int((z_max-z_min)/segLength), 1)#2 meter segments but at least 1
			
			#fit piecewise linear lines and color them by their slope
			lines_rotated_temp = []
			line_colors_temp = []
			for j in range(numSlopes):
				z1 = (j/numSlopes)*(z_max-z_min)+z_min
				z2 = ((j+1)/numSlopes)*(z_max-z_min)+z_min

				slope, intercept, y, line, line_color = fitLineZ1toZ2(points_2d, z1, z2, v1_rotated[0], v2_rotated[0], thresh)
				if slope is None:
					continue
				line_unrotated = np.dot(line - v1, inverse_rotation_matrix.T) + v1
				lines.append(line_unrotated)
				line_colors_temp.append(line_color)
				lines_rotated_temp.append(line)

			if len(lines_rotated_temp)>0:
				lines_rotated.append(np.vstack(lines_rotated_temp))
				line_colors.append(np.vstack(line_colors_temp))

			slope, intercept, y, line, line_color = fitLineZ1toZ2(points_2d, z_min, z_max, v1_rotated[0], v2_rotated[0], thresh)

			if slope is None:
				continue
			y_ref = slope*z_min+intercept
			
			#New slope algorithm:
			top18_ind = np.searchsorted(points_2d[:,1], z_max-0.45)#top 18 inches of points below z_max
			avg_y = np.mean(points_2d[top18_ind:,0])
			delta_y = (avg_y-y_ref)
			delta_z = z_max - z_min
			new_slope = (delta_y / delta_z)
			new_slopes.append(new_slope)
			new_slope_color = np.tile(value_to_rgb_jet(new_slope), (line.shape[0], 1))
			new_slope_colors.append(new_slope_color)

			# Convert to RGB using jet colormap
			cmap = plt.cm.jet

			slope_values.append(slope)

			pc_slices_rotated.append(pc_slice)

			pc_slice_unrotated = np.dot(pc_slice - v1, inverse_rotation_matrix.T) + v1
			line_unrotated = np.dot(line - v1, inverse_rotation_matrix.T) + v1

			#Color original points by their distance from the reference point at the base
			pc_slice_deltas = -(pc_slice[:, 1] - y_ref) / 0.3#0.35m max displacement for colors
			colors_pc_slice = cmap(pc_slice_deltas)[:, :3]

			pc_slices.append(pc_slice_unrotated)
			pc_slice_colors.append(colors_pc_slice)

		pc_slices_unrolled = unrollSlices(pc_slices_rotated, spacing)#spacing = 1 meter
		lines_unrolled = unrollSlices(lines_rotated, spacing)#10 lines per spacing

		pc_slices = np.vstack(pc_slices)
		pc_slice_colors = np.vstack(pc_slice_colors)
		lines = np.vstack(lines)
		line_colors = np.vstack(line_colors)
		slope_values = np.vstack(slope_values)


		new_slope_colors = np.vstack(new_slope_colors)

		points_temp = []
		colors_temp = []
		for k in range(len(pc_slices_rotated)):
			points, colors = getCrossSection(pc_slices_rotated[k], lines_rotated[k], "pointClouds/crossSections/wall_%d/section_%d.ply"%(wall_id,k), k, value_to_rgb_jet(slope_values[k][0])[:3])
			points_temp.append(points)
			colors_temp.append(colors)
		points = np.vstack(points_temp)
		colors = np.vstack(colors_temp)
		savePoints(points,"pointClouds/unrolled/crossSections/wall_%d.ply"%wall_id,colors=colors)

		if thresh is None:
			savePoints(pc_slices,"pointClouds/displacements/pc_slice_%d_%.1f.ply"%(wall_id,spacing), colors=pc_slice_colors)
			savePoints(lines,"pointClouds/slopes/line_%d_%.1f.ply"%(wall_id,spacing), colors=line_colors)
			np.savetxt("renders/slopes/slope_%d_%.1f.csv"%(wall_id,spacing), slope_values, delimiter=",", fmt='%.6f')
			savePoints(lines,"pointClouds/new_slopes/line_%d_%.1f.ply"%(wall_id,spacing), colors=new_slope_colors)

			savePoints(pc_slices_unrolled, "pointClouds/unrolled/displacements/pc_slices_unrolled_%d_%.1f.ply"%(wall_id,spacing), colors=pc_slice_colors)
			savePoints(lines_unrolled, "pointClouds/unrolled/slopes/lines_unrolled_%d_%.1f.ply"%(wall_id,spacing), colors=line_colors)
			savePoints(lines_unrolled,"pointClouds/unrolled/new_slopes/line_%d_%.1f.ply"%(wall_id,spacing), colors=new_slope_colors)
		else:
			savePoints(lines_unrolled, "pointClouds/unrolled/slopes/lines_unrolled_%d_%3.3f_%.1f.ply"%(wall_id,thresh,spacing), colors=line_colors)
			