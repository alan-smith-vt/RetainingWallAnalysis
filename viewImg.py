import cv2
import matplotlib.pyplot as plt

def display_image(image):    
    # Check if image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert from BGR (OpenCV default) to RGB (matplotlib default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create matplotlib figure and display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.title('Image Display using OpenCV and Matplotlib')
    plt.tight_layout()
    plt.show()

#Load image
wall_id = 1

image_path = 'pdfs/wall_%d.png'%wall_id
image = cv2.imread(image_path)

# Calculate scale factor: 305/328 ≈ 0.9299
if wall_id == 1:
    scale_factor = 305 / 328
elif wall_id == 2:
    scale_factor = 305 / 418
elif wall_id == 3:
    scale_factor = 305 / 446

# Get original dimensions
height, width = image.shape[:2]

# Calculate new dimensions
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Save the result
cv2.imwrite('renders/wall_%d_dwg.png'%wall_id, resized_image)

print(f"Original size: {width} x {height}")
print(f"New size: {new_width} x {new_height}")
print(f"Scale factor: {scale_factor:.4f}")

display_image(resized_image)
display_image(cv2.imread("renders/displacements/%d.png"%wall_id))