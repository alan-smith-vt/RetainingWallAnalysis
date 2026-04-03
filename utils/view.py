"""
Image viewer and wall drawing rescaler.

Rescales engineering wall drawings to match the analysis pixel scale
(305 px = 10 ft) and displays them for verification.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import matplotlib.pyplot as plt
from config import WALL_SCALE_FACTORS


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def display_image(image):
    if image is None:
        print("Error: Could not load image")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Image Display using OpenCV and Matplotlib')
    plt.tight_layout()
    plt.show()


# ── Main execution ──────────────────────────────────────────────────────────

wall_id = 1

image_path = 'pdfs/wall_%d.png' % wall_id
image = cv2.imread(image_path)

scale_factor = WALL_SCALE_FACTORS[wall_id]

height, width = image.shape[:2]

new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

ensure_dir('outputs/images/wall_%d_dwg.png' % wall_id)
cv2.imwrite('outputs/images/wall_%d_dwg.png' % wall_id, resized_image)

print(f"Original size: {width} x {height}")
print(f"New size: {new_width} x {new_height}")
print(f"Scale factor: {scale_factor:.4f}")

display_image(resized_image)
display_image(cv2.imread("outputs/images/displacement_%d.png" % wall_id))
