"""
Grid and elevation label overlay.

Adds structural grid lines (station markers and elevation labels) to
elevation profile images, draws a red reference line, and composites
settlement curve overlays.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import glob
from tqdm import tqdm
from datetime import datetime
from config import (
    GRID_PAD_X, GRID_PADDING, GRID_LINE_THICKNESS, GRID_DELTA_X,
    GRID_DELTA_Y, GRID_FONT_SIZE, GRID_RED_LINE_ELEVATION,
    GRID_RED_LINE_THICKNESS,
)


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def printf(msg):
    print("[%s]: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def get_font(font_size=20):
    """Helper function to get a font object."""
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
    return font


def add_all_text_to_image(image, vertical_texts, horizontal_texts, font_size=20, color=(255, 255, 255)):
    """
    Add all text (vertical and horizontal) to image in one PIL conversion.
    """
    if len(image.shape) == 3:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_image = Image.fromarray(image)

    draw = ImageDraw.Draw(pil_image)
    font = get_font(font_size)
    width, height = pil_image.size

    for text, x, y in vertical_texts:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = 20
        temp_width = text_width + padding * 2
        temp_height = text_height + padding * 2

        text_image = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text((padding, padding), text, font=font, fill=color + (255,))
        rotated_text = text_image.rotate(90, expand=True)

        rotated_width, rotated_height = rotated_text.size
        final_x = max(0, min(x, width - 1))
        final_y = max(0, min(y, height - 1))

        paste_width = min(rotated_width, width - final_x)
        paste_height = min(rotated_height, height - final_y)

        if paste_width > 0 and paste_height > 0:
            if paste_width < rotated_width or paste_height < rotated_height:
                rotated_text = rotated_text.crop((0, 0, paste_width, paste_height))
            pil_image.paste(rotated_text, (final_x, final_y), rotated_text)

    for text, x, y in horizontal_texts:
        final_x = max(0, min(x, width - 1))
        final_y = max(0, min(y, height - 1))
        draw.text((final_x, final_y), text, font=font, fill=color)

    result_array = np.array(pil_image)
    if len(image.shape) == 3:
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
    else:
        return result_array


def overlay_images_white_mask(base_image, overlay_image, x_offset, y_offset, white_threshold=240):
    """Overlay an image where white pixels in the overlay are transparent."""
    base_h, base_w = base_image.shape[:2]
    overlay_h, overlay_w = overlay_image.shape[:2]

    result = base_image.copy()

    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(base_w, x_offset + overlay_w)
    y_end = min(base_h, y_offset + overlay_h)

    overlay_x_start = max(0, -x_offset)
    overlay_y_start = max(0, -y_offset)
    overlay_x_end = overlay_x_start + (x_end - x_start)
    overlay_y_end = overlay_y_start + (y_end - y_start)

    if x_start >= x_end or y_start >= y_end:
        print("Warning: No overlap between images with given offset")
        return result

    base_region = result[y_start:y_end, x_start:x_end]
    overlay_region = overlay_image[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]

    if len(overlay_region.shape) == 3:
        white_mask = np.all(overlay_region >= white_threshold, axis=2)
    else:
        white_mask = overlay_region >= white_threshold

    non_white_mask = ~white_mask

    if len(overlay_region.shape) == 3:
        non_white_mask_3d = np.repeat(non_white_mask[:, :, np.newaxis], 3, axis=2)
        base_region[non_white_mask_3d] = overlay_region[non_white_mask_3d]
    else:
        base_region[non_white_mask] = overlay_region[non_white_mask]

    result[y_start:y_end, x_start:x_end] = base_region

    return result


# ── Main execution ──────────────────────────────────────────────────────────

files = glob.glob("renders/elevations/*.png")
printf("Processing elevation images with grid overlays")

for file in files:
    wall_id = file.split("\\")[-1][0]
    img = cv2.imread(file)

    padx = GRID_PAD_X
    padding = GRID_PADDING

    thickness = GRID_LINE_THICKNESS
    delta_x = GRID_DELTA_X
    delta_y = GRID_DELTA_Y

    new_img = np.ones((img.shape[0] + padding[0] + padding[1], img.shape[1] + padding[2] + padding[3], 3), np.uint8) * 255
    new_img[padding[0]:img.shape[0] + padding[0], padding[2]:img.shape[1] + padding[2]] = img

    vertical_texts = []
    horizontal_texts = []

    printf(f"Drawing grid lines for Wall #{wall_id}")

    for i in range(2, img.shape[1] // delta_x + 3):
        pos_pix = i * delta_x
        pos_ft = i * delta_x * 10 / 305 - 20
        new_img[:img.shape[0] + 100, i * delta_x - thickness:i * delta_x + thickness] = 0

        text = "%d+%02d" % (pos_ft // 100, pos_ft % 100)
        vertical_texts.append((text, pos_pix - 75, 1400))

    for i in range(1, int(img.shape[0] / delta_y) + 1):
        pos_pix = i * delta_y
        pos_ft = i * delta_y * 1 / 610
        new_img[int(i * delta_y) - thickness:int(i * delta_y) + thickness, padx - 100:] = 0

        text = "%3.2f" % (762.5 - pos_ft)
        horizontal_texts.append((text, 75, int(pos_pix) - 75))

    # Draw red line at reference elevation
    pos_pix = int((1219 / 2) * (762.5 - GRID_RED_LINE_ELEVATION))
    red_thickness = GRID_RED_LINE_THICKNESS
    new_img[pos_pix - red_thickness:pos_pix + red_thickness, padx - 100:] = 0
    new_img[pos_pix - red_thickness:pos_pix + red_thickness, padx - 100:, 2] = 255

    printf(f"Adding all text labels for Wall #{wall_id} in one pass")
    new_img = add_all_text_to_image(
        new_img,
        vertical_texts,
        horizontal_texts,
        font_size=GRID_FONT_SIZE,
        color=(0, 0, 0)
    )

    overlay_img = cv2.imread("renders/elevations/curves/%s_elevation.png" % wall_id)
    result_image = overlay_images_white_mask(new_img, overlay_img, padx, 0)
    ensure_dir("renders/overlays/elevations/%s.png" % wall_id)
    cv2.imwrite("renders/overlays/elevations/%s.png" % wall_id, result_image)
    printf(f"Completed Wall #{wall_id}")

printf("All walls processed successfully!")
