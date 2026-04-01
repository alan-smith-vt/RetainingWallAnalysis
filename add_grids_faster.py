import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import glob
from tqdm import tqdm
from datetime import datetime

def printf(msg):
    print("[%s]: %s"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))

def get_font(font_size=20):
    """Helper function to get a font object"""
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)  # macOS
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)  # Linux
            except:
                font = ImageFont.load_default()
    return font

def add_all_text_to_image(image, vertical_texts, horizontal_texts, font_size=20, color=(255, 255, 255)):
    """
    Add all text (vertical and horizontal) to image in one PIL conversion.
    
    Args:
        image: Input image (numpy array)
        vertical_texts: List of tuples (text, x, y) for vertical text
        horizontal_texts: List of tuples (text, x, y) for horizontal text
        font_size: Size of the font
        color: Text color in RGB format
    
    Returns:
        Modified image as numpy array
    """
    # Convert OpenCV image to PIL once
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Get font once
    font = get_font(font_size)
    
    width, height = pil_image.size
    
    # Add all vertical texts
    for text, x, y in vertical_texts:
        # Get text dimensions with proper padding
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add padding to ensure text isn't cropped
        padding = 20
        temp_width = text_width + padding * 2
        temp_height = text_height + padding * 2
        
        # Create a temporary image for the text
        text_image = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_image)
        
        # Draw text with padding offset
        text_draw.text((padding, padding), text, font=font, fill=color + (255,))
        
        # Rotate the text 90 degrees counterclockwise
        rotated_text = text_image.rotate(90, expand=True)
        
        # Get rotated dimensions
        rotated_width, rotated_height = rotated_text.size
        
        # Ensure we don't go outside image bounds
        final_x = max(0, min(x, width - 1))
        final_y = max(0, min(y, height - 1))
        
        # Crop rotated text if it extends beyond image boundaries
        paste_width = min(rotated_width, width - final_x)
        paste_height = min(rotated_height, height - final_y)
        
        if paste_width > 0 and paste_height > 0:
            # Crop the rotated text if necessary
            if paste_width < rotated_width or paste_height < rotated_height:
                rotated_text = rotated_text.crop((0, 0, paste_width, paste_height))
            
            # Paste the rotated text onto the main image
            pil_image.paste(rotated_text, (final_x, final_y), rotated_text)
    
    # Add all horizontal texts
    for text, x, y in horizontal_texts:
        # Ensure we don't go outside image bounds
        final_x = max(0, min(x, width - 1))
        final_y = max(0, min(y, height - 1))
        
        # Draw text directly
        draw.text((final_x, final_y), text, font=font, fill=color)
    
    # Convert back to OpenCV format once at the end
    result_array = np.array(pil_image)
    if len(image.shape) == 3:
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
    else:
        return result_array

def pad_image(img, padding):
    new_img = np.ones((img.shape[0]+padding[0]+padding[1], img.shape[1]+padding[2]+padding[3],3))*255
    new_img[padding[0]:img.shape[1], padding[2]:img.shape[0]] = img
    return new_img


def overlay_images_white_mask(base_image, overlay_image, x_offset, y_offset, white_threshold=240):
    """
    Overlay an image where white pixels in the overlay are transparent.
    
    Parameters:
    base_image: Base image (numpy array)
    overlay_image: Overlay image (numpy array)  
    x_offset, y_offset: Position to place overlay
    white_threshold: Pixel values above this are considered "white" (0-255 scale)
    
    Returns:
    Result image with overlay applied, white pixels from overlay ignored
    """
    
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
    
    # Extract the regions
    base_region = result[y_start:y_end, x_start:x_end]
    overlay_region = overlay_image[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
    
    # Create mask for non-white pixels
    if len(overlay_region.shape) == 3:  # Color image
        # For color images, check if ALL channels are above threshold
        white_mask = np.all(overlay_region >= white_threshold, axis=2)
    else:  # Grayscale image
        white_mask = overlay_region >= white_threshold
    
    # Invert mask - we want to keep non-white pixels
    non_white_mask = ~white_mask
    
    # Apply the overlay only where pixels are not white
    if len(overlay_region.shape) == 3:  # Color image
        # Expand mask to match color channels
        non_white_mask_3d = np.repeat(non_white_mask[:, :, np.newaxis], 3, axis=2)
        base_region[non_white_mask_3d] = overlay_region[non_white_mask_3d]
    else:  # Grayscale image
        base_region[non_white_mask] = overlay_region[non_white_mask]
    
    # Place the modified region back into the result image
    result[y_start:y_end, x_start:x_end] = base_region
    
    return result

# Main processing
files = glob.glob("renders/elevations/*.png")
printf("Optimized version: Single PIL conversion per image for all text operations")

for file in files:
    wall_id = file.split("\\")[-1][0]  # number of wall
    img = cv2.imread(file)
    
    # y top, y bottom, x left, x right
    padx = 305*2  # 2 stations
    padding = [0, 900, padx, 10]
    
    thickness = 3
    delta_x = 305  # 10 ft
    # 1219 pixels per 2 ft
    delta_y = (1219/24)*3  # try 3 in spacing
    
    # Create padded image
    new_img = np.ones((img.shape[0]+padding[0]+padding[1], img.shape[1]+padding[2]+padding[3], 3), np.uint8)*255
    new_img[padding[0]:img.shape[0]+padding[0], padding[2]:img.shape[1]+padding[2]] = img
    
    # Collect all text to be added
    vertical_texts = []
    horizontal_texts = []
    
    # FIRST LOOP: Draw all grid lines (stays in numpy for efficiency)
    printf(f"Drawing grid lines for Wall #{wall_id}")
    
    # Draw vertical grid lines
    for i in range(2, img.shape[1]//delta_x+3):
        pos_pix = i*delta_x
        pos_ft = i*delta_x*10/305-20  # 305 pixels per 10 ft
        # Draw vertical line
        new_img[:img.shape[0]+100, i*delta_x-thickness:i*delta_x+thickness] = 0
        
        # Prepare vertical text (but don't draw yet)
        text = "%d+%02d"%(pos_ft//100, pos_ft%100)
        vertical_texts.append((text, pos_pix-75, 1400))
    
    # Draw horizontal grid lines
    for i in range(1, int(img.shape[0]/delta_y)+1):
        pos_pix = i*delta_y
        pos_ft = i*delta_y*1/610  # 305 pixels per 1 ft
        # Draw horizontal line
        new_img[int(i*delta_y)-thickness:int(i*delta_y)+thickness, padx-100:] = 0
        
        # Prepare horizontal text (but don't draw yet)
        text = "%3.2f"%(762.5-pos_ft)
        horizontal_texts.append((text, 75, int(pos_pix)-75))

    #Draw red line @ 761.6
    pos_pix = int((1219/2)*(762.5-761.6))#offset to reach the 761.6 ft elevation
    # Draw horizontal line
    thickness = 6
    new_img[pos_pix-thickness:pos_pix+thickness, padx-100:] = 0
    new_img[pos_pix-thickness:pos_pix+thickness, padx-100:,2] = 255
    
    # SECOND STEP: Add all text at once with single PIL conversion
    printf(f"Adding all text labels for Wall #{wall_id} in one pass")
    new_img = add_all_text_to_image(
        new_img, 
        vertical_texts, 
        horizontal_texts, 
        font_size=100, 
        color=(0, 0, 0)
    )
    
    # Save the result
    overlay_img = cv2.imread("renders/elevations/curves/%s_elevation.png"%wall_id)
    result_image = overlay_images_white_mask(new_img, overlay_img, padx, 0)
    cv2.imwrite("renders/overlays/elevations/%s.png"%wall_id, result_image)
    printf(f"Completed Wall #{wall_id}")

printf("All walls processed successfully!")