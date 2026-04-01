"""
Engineering drawing overlay.

Composites analysis results (elevation profiles, slopes, displacements)
onto the corresponding engineering wall drawings with configurable
transparency and per-wall pixel offsets.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from config import OVERLAY_TRANSPARENCY, WALL_OVERLAY_OFFSETS


def overlay_images(base_image_path, overlay_image_path, x_offset, y_offset, transparency=None):
    """
    Overlay two images with transparency and offset.

    Parameters
    ----------
    base_image_path : str — path to the base/background image
    overlay_image_path : str — path to the overlay image
    x_offset : int — horizontal offset in pixels
    y_offset : int — vertical offset in pixels
    transparency : float — overlay opacity 0.0–1.0 (default from config)

    Returns
    -------
    numpy.ndarray — combined image
    """
    if transparency is None:
        transparency = OVERLAY_TRANSPARENCY

    base_image = cv2.imread(base_image_path)
    overlay_image = cv2.imread(overlay_image_path)

    if base_image is None:
        print(f"Error: Could not load base image from {base_image_path}")
        return None

    if overlay_image is None:
        print(f"Error: Could not load overlay image from {overlay_image_path}")
        return None

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

    alpha = transparency
    blended = cv2.addWeighted(base_region, 1 - alpha, overlay_region, alpha, 0)

    result[y_start:y_end, x_start:x_end] = blended

    return result


# ── Main execution ──────────────────────────────────────────────────────────

for wall_id in [1, 2, 3]:
    scale = "0.1"
    base_path = "renders/wall_%d_dwg.png" % wall_id
    overlay_path = "renders/elevations/%d_%s_elevation.png" % (wall_id, scale)

    base_img = cv2.imread(base_path)
    overlay_img = cv2.imread(overlay_path)

    base_y = base_img.shape[0]
    overlay_y = overlay_img.shape[0]

    x_offset, y_delta = WALL_OVERLAY_OFFSETS[wall_id]
    y_offset = (base_y - overlay_y) - y_delta

    result_image = overlay_images(base_path, overlay_path, x_offset, y_offset)

    if result_image is not None:
        cv2.imwrite("renders/overlays/elevations/wall_%d_overlay.jpg" % wall_id, result_image)
        print("Wall %d, Overlaid image saved" % wall_id)
