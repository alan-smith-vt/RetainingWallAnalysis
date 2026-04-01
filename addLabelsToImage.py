import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import glob

def add_vertical_text_pil(image, text_list, font_size=20, color=(255, 255, 255)):
    """
    Add vertically oriented text using PIL (better text rendering and rotation support).
    
    Args:
        image: Input image (numpy array or PIL Image)
        text_list: List of strings to be placed as vertical text
        font_size: Size of the font
        color: Text color in RGB format
    
    Returns:
        Modified image as numpy array
    """
    # Convert OpenCV image to PIL if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a better font, fall back to default if not available
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
    
    width, height = pil_image.size
    
    for i, text in enumerate(text_list):
        # Calculate x position (100 pixels spacing)
        x_pos = 50 + i * 100
        
        # Skip if x position exceeds image width (more lenient check)
        if x_pos >= width - 10:  # Minimal margin
            break
        
        # Get text dimensions with proper padding
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add padding to ensure text isn't cropped
        padding = 10
        temp_width = text_width + padding * 2
        temp_height = text_height + padding * 2
        
        # Create a temporary image for the text (larger to prevent cropping)
        text_image = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_image)
        
        # Draw text with padding offset
        text_draw.text((padding, padding), text, font=font, fill=color + (255,))
        
        # Rotate the text 90 degrees counterclockwise
        rotated_text = text_image.rotate(90, expand=True)
        
        # Calculate position to center the rotated text vertically
        rotated_width, rotated_height = rotated_text.size
        y_pos = (height - rotated_height) // 2
        
        # Adjust x position to account for rotation
        final_x = x_pos - rotated_width // 2
        
        # Ensure we don't go outside image bounds (more lenient)
        final_x = max(0, min(final_x, width - min(rotated_width, width)))
        y_pos = max(0, min(y_pos, height - min(rotated_height, height)))
        
        # Paste the rotated text onto the main image
        pil_image.paste(rotated_text, (final_x, y_pos), rotated_text)
    
    # Convert back to OpenCV format
    if isinstance(image, np.ndarray):
        result_array = np.array(pil_image)
        if len(image.shape) == 3:
            return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        else:
            return result_array
    else:
        return pil_image

files = glob.glob("renders/slopes/*.png")
for file in files:
    #pattern = r'\\([^\\.]*)\.(?:[^.]*)?$'
    #name = re.search(pattern, file).group(1)
    name = file[-9:-4]
    source_image = cv2.imread(file)
    text_labels = np.loadtxt("renders/slopes/slope_" + name + ".csv", delimiter=',')
    text_labels = np.round(text_labels*100,2)
    text_labels = text_labels.astype(str)
    text_labels = np.char.add(text_labels, '%')
    text_labels = np.append(text_labels, '')
    
    # Method 2: Using PIL (better text rotation)
    result_pil = add_vertical_text_pil(source_image, text_labels, font_size=64, color=(100, 100, 100))
    cv2.imwrite('renders/slopes/labeled/slope_%s_labeled.png'%name, result_pil)
    
    print("Vertical text images created successfully!")
    #print(f"Text positions: {[50 + i * 100 for i in range(len(text_labels))]}")