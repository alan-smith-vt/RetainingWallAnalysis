"""
Scale reference template generator.

Creates a solid black square image used as a scale reference in visualizations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from config import SCALE_TEMPLATE_SIZE

def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


width = SCALE_TEMPLATE_SIZE
height = SCALE_TEMPLATE_SIZE

black_image = np.zeros((height, width, 3), dtype=np.uint8)
ensure_dir('black_image_%dx%d.png' % (width, height))
cv2.imwrite('black_image_%dx%d.png' % (width, height), black_image)
print(f"Created black image: {width}x{height} pixels")

black_gray = np.zeros((height, width), dtype=np.uint8)
ensure_dir('black_image_%dx%d_gray.png' % (width, height))
cv2.imwrite('black_image_%dx%d_gray.png' % (width, height), black_gray)
print(f"Created grayscale black image: {width}x{height} pixels")
