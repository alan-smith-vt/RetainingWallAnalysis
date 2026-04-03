"""
Image stacker.

Vertically concatenates per-wall elevation overlay images into a single
combined image, padding narrower images to match the widest.
"""

import os
import cv2
import numpy as np


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

# ── Main execution ──────────────────────────────────────────────────────────

imgs = []
for wall_id in [1, 2, 3]:
    file = "outputs/images/elevation_overlay_%d.png" % wall_id
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    imgs.append(img)

# Find max width
max_width = max(img.shape[1] for img in imgs)

# Pad images to max width with transparent pixels
padded_imgs = []
for img in imgs:
    if img.shape[1] < max_width:
        pad = max_width - img.shape[1]
        pad_value = (0, 0, 0, 0) if img.shape[2] == 4 else (0, 0, 0)
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=pad_value)
    padded_imgs.append(img)

# Stack and save
stacked = cv2.vconcat(padded_imgs)
ensure_dir("outputs/images/elevation_overlay_combined.png")
cv2.imwrite("outputs/images/elevation_overlay_combined.png", stacked)
