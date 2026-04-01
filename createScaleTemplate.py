import cv2
import numpy as np

# Method 1: Create black image using np.zeros (most common)
def create_black_image_zeros(width, height):
    # For grayscale: shape = (height, width)
    # For color (BGR): shape = (height, width, 3)
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    return black_image

# Method 2: Create black image using np.full
def create_black_image_full(width, height):
    black_image = np.full((height, width, 3), 0, dtype=np.uint8)
    return black_image

# Method 3: Create and explicitly set to black
def create_black_image_explicit(width, height):
    black_image = np.empty((height, width, 3), dtype=np.uint8)
    black_image[:] = [0, 0, 0]  # BGR format
    return black_image

# Create 305x305 black image
width = 305
height = 305

# Using the most common method
black_image = np.zeros((height, width, 3), dtype=np.uint8)

# Save the image
cv2.imwrite('black_image_305x305.png', black_image)
print(f"Created black image: {width}x{height} pixels")
print(f"Image shape: {black_image.shape}")
print(f"Data type: {black_image.dtype}")

# Alternative: Create grayscale black image (single channel)
black_gray = np.zeros((height, width), dtype=np.uint8)
cv2.imwrite('black_image_305x305_gray.png', black_gray)
print(f"Created grayscale black image: {width}x{height} pixels")

# Verify the image properties
print(f"Color image shape: {black_image.shape}")  # Should be (305, 305, 3)
print(f"Grayscale image shape: {black_gray.shape}")  # Should be (305, 305)
print(f"All pixels are black (0): {np.all(black_image == 0)}")

# Optional: Display the image (if running interactively)
# cv2.imshow('Black Image', black_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# One-liner version for quick use:
# black_img = np.zeros((305, 305, 3), dtype=np.uint8)