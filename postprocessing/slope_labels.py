"""
Slope percentage label overlay.

Reads slope CSV data and overlays vertical percentage labels onto the
corresponding slope visualization images.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import glob
from config import SLOPE_LABEL_FONT_SIZE, SLOPE_LABEL_COLOR


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def add_vertical_text_pil(image, text_list, font_size=20, color=(255, 255, 255)):
    """Add vertically oriented text using PIL."""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
    else:
        pil_image = image

    draw = ImageDraw.Draw(pil_image)

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

    width, height = pil_image.size

    for i, text in enumerate(text_list):
        x_pos = 50 + i * 100

        if x_pos >= width - 10:
            break

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = 10
        temp_width = text_width + padding * 2
        temp_height = text_height + padding * 2

        text_image = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_draw.text((padding, padding), text, font=font, fill=color + (255,))
        rotated_text = text_image.rotate(90, expand=True)

        rotated_width, rotated_height = rotated_text.size
        y_pos = (height - rotated_height) // 2
        final_x = x_pos - rotated_width // 2

        final_x = max(0, min(final_x, width - min(rotated_width, width)))
        y_pos = max(0, min(y_pos, height - min(rotated_height, height)))

        pil_image.paste(rotated_text, (final_x, y_pos), rotated_text)

    if isinstance(image, np.ndarray):
        result_array = np.array(pil_image)
        if len(image.shape) == 3:
            return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        else:
            return result_array
    else:
        return pil_image


# ── Main execution ──────────────────────────────────────────────────────────

files = glob.glob("outputs/images/slope_*.png")
for file in files:
    basename = os.path.splitext(os.path.basename(file))[0]  # e.g. slope_1_1.0
    # Skip already-labeled files and threshold files
    if "_labeled" in basename or "threshold" in basename:
        continue
    name = basename.replace("slope_", "")  # e.g. 1_1.0
    source_image = cv2.imread(file)
    text_labels = np.loadtxt("outputs/images/slope_%s.csv" % name, delimiter=',')
    text_labels = np.round(text_labels * 100, 2)
    text_labels = text_labels.astype(str)
    text_labels = np.char.add(text_labels, '%')
    text_labels = np.append(text_labels, '')

    result_pil = add_vertical_text_pil(source_image, text_labels, font_size=SLOPE_LABEL_FONT_SIZE, color=SLOPE_LABEL_COLOR)
    ensure_dir('outputs/images/slope_%s_labeled.png' % name)
    cv2.imwrite('outputs/images/slope_%s_labeled.png' % name, result_pil)

    print("Vertical text images created successfully!")
