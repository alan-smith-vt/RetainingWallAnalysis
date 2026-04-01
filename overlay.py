import cv2
import numpy as np

def overlay_images(base_image_path, overlay_image_path, x_offset, y_offset, transparency=0.5):
    """
    Overlay two images with transparency and offset
    
    Args:
        base_image_path (str): Path to the base/background image
        overlay_image_path (str): Path to the overlay image
        x_offset (int): Horizontal offset in pixels
        y_offset (int): Vertical offset in pixels
        transparency (float): Transparency of overlay image (0.0 to 1.0)
    
    Returns:
        numpy.ndarray: Combined image
    """
    # Load both images
    base_image = cv2.imread(base_image_path)
    overlay_image = cv2.imread(overlay_image_path)
    
    if base_image is None:
        print(f"Error: Could not load base image from {base_image_path}")
        return None
    
    if overlay_image is None:
        print(f"Error: Could not load overlay image from {overlay_image_path}")
        return None
    
    # Get dimensions
    base_h, base_w = base_image.shape[:2]
    overlay_h, overlay_w = overlay_image.shape[:2]
    
    # Create a copy of the base image
    result = base_image.copy()
    
    # Calculate the region where overlay will be placed
    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(base_w, x_offset + overlay_w)
    y_end = min(base_h, y_offset + overlay_h)
    
    # Calculate corresponding region in overlay image
    overlay_x_start = max(0, -x_offset)
    overlay_y_start = max(0, -y_offset)
    overlay_x_end = overlay_x_start + (x_end - x_start)
    overlay_y_end = overlay_y_start + (y_end - y_start)
    
    # Check if there's any overlap
    if x_start >= x_end or y_start >= y_end:
        print("Warning: No overlap between images with given offset")
        return result
    
    # Extract the regions to blend
    base_region = result[y_start:y_end, x_start:x_end]
    overlay_region = overlay_image[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
    
    # Blend the images using weighted addition
    # result = base * (1 - alpha) + overlay * alpha
    alpha = transparency
    blended = cv2.addWeighted(base_region, 1 - alpha, overlay_region, alpha, 0)
    
    # Place the blended region back into the result image
    result[y_start:y_end, x_start:x_end] = blended
    
    return result
'''
for wall_id in [1,2,3]:
    for target in ["displacements","slopes"]:
        if target == "displacements":
            scale = "0.1"
        elif target == "slopes":
            scale = "1.0"
        base_path = "renders/wall_%d_dwg.png"%wall_id
        overlay_path = "renders/%s/%d_%s.png"%(target,wall_id, scale)

        base_img = cv2.imread(base_path)
        overlay_img = cv2.imread(overlay_path)

        base_y = base_img.shape[0]
        overlay_y = overlay_img.shape[0]

        if wall_id==1:
            x_offset = 180 # pixels to the right
            y_offset = (base_y-overlay_y)-600 # pixels up
        elif wall_id==2:
            x_offset = 180 # pixels to the right
            y_offset = (base_y-overlay_y)-444 # pixels up
        elif wall_id==3:
            x_offset = 150 # pixels to the right
            y_offset = (base_y-overlay_y)-414 # pixels up
            
        result_image = overlay_images(base_path, overlay_path, x_offset, y_offset, transparency=0.5)
        
        if result_image is not None:
            # Save the result
            cv2.imwrite("renders/overlays/wall_%d_%s_overlay.jpg"%(wall_id,target), result_image)
            print("%s, Wall %d, Overlaid image saved"%(target, wall_id))
'''
'''
for wall_id in [1,2,3]:
    for thresh in [-0.01, -0.015, -0.02, -0.025, -0.03, -0.035]:
        scale = "1.0"
        base_path = "renders/wall_%d_dwg.png"%wall_id
        overlay_path = "renders/slopes/thresholds/%d_%3.3f_%s.png"%(wall_id, thresh, scale)

        base_img = cv2.imread(base_path)
        overlay_img = cv2.imread(overlay_path)

        base_y = base_img.shape[0]
        overlay_y = overlay_img.shape[0]

        if wall_id==1:
            x_offset = 180 # pixels to the right
            y_offset = (base_y-overlay_y)-600 # pixels up
        elif wall_id==2:
            x_offset = 180 # pixels to the right
            y_offset = (base_y-overlay_y)-444 # pixels up
        elif wall_id==3:
            x_offset = 150 # pixels to the right
            y_offset = (base_y-overlay_y)-414 # pixels up
            
        result_image = overlay_images(base_path, overlay_path, x_offset, y_offset, transparency=0.5)
        
        if result_image is not None:
            # Save the result
            cv2.imwrite("renders/overlays/thresholds/wall_%d_%3.3f_overlay.jpg"%(wall_id,thresh), result_image)
            print("%3.3f, Wall %d, Overlaid image saved"%(thresh, wall_id))
'''

for wall_id in [1,2,3]:
    scale = "0.1"
    base_path = "renders/wall_%d_dwg.png"%wall_id
    overlay_path = "renders/elevations/%d_%s_elevation.png"%(wall_id, scale)

    base_img = cv2.imread(base_path)
    overlay_img = cv2.imread(overlay_path)

    base_y = base_img.shape[0]
    overlay_y = overlay_img.shape[0]

    if wall_id==1:
        x_offset = 180 # pixels to the right
        y_offset = (base_y-overlay_y)-416 # pixels up
    elif wall_id==2:
        x_offset = 180 # pixels to the right
        y_offset = (base_y-overlay_y)-319 # pixels up
    elif wall_id==3:
        x_offset = 150 # pixels to the right
        y_offset = (base_y-overlay_y)-308 # pixels up
        
    result_image = overlay_images(base_path, overlay_path, x_offset, y_offset, transparency=0.5)
    
    if result_image is not None:
        # Save the result
        cv2.imwrite("renders/overlays/elevations/wall_%d_overlay.jpg"%(wall_id), result_image)
        print("Wall %d, Overlaid image saved"%(wall_id))