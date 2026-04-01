import cv2
import numpy as np
from glob import glob

# for thresh in [-0.01, -0.015, -0.02, -0.025, -0.03, -0.035]:
# 	files = glob("renders/overlays/thresholds/wall_*_%3.3f_overlay.jpg"%thresh)
# 	# Stack vertically
# 	imgs = []
# 	for file in files:
# 		img = cv2.imread(file)
# 		imgs.append(img)

# 	# Find max width
# 	max_width = max(img.shape[1] for img in imgs)

# 	# Pad images to max width with white
# 	padded_imgs = []
# 	for img in imgs:
# 		if img.shape[1] < max_width:
# 			pad = max_width - img.shape[1]
# 			img = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
# 		padded_imgs.append(img)

# 	# Stack and save
# 	stacked = cv2.vconcat(padded_imgs)
# 	cv2.imwrite("renders/overlays/thresh_%3.3f.png"%thresh, stacked)

# Stack vertically
imgs = []
for wall_id in [1,2,3]:
	file = "renders/overlays/elevations/%d.png"%wall_id
	img = cv2.imread(file)
	imgs.append(img)

# Find max width
max_width = max(img.shape[1] for img in imgs)

# Pad images to max width with white
padded_imgs = []
for img in imgs:
	if img.shape[1] < max_width:
		pad = max_width - img.shape[1]
		img = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
	padded_imgs.append(img)

# Stack and save
stacked = cv2.vconcat(padded_imgs)
cv2.imwrite("renders/overlays/elevations/combined.png", stacked)