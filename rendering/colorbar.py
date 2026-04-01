"""
Colorbar / legend generator.

Creates reference legend images for slope percentage and displacement
color scales used in the analysis visualizations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from config import SLOPE_COLORMAP_RANGE


def hex_to_rgb(hex_color):
    """Convert hex color to RGB array."""
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def value_to_rgb(percent):
    if percent > 0:
        return np.array([0.0, 0.0, 0.0, 0.5])
    cmap = plt.cm.jet
    percent = (-percent * 100) / SLOPE_COLORMAP_RANGE
    percent = min(1.0, percent)
    rgba = cmap(percent)
    return np.array([rgba[0], rgba[1], rgba[2], 0.5])


def displacement_to_rgb(inches):
    """Map displacement values (0 to 12 inches) to jet colormap with 50% transparency."""
    if inches < 0:
        inches = 0
    elif inches > 12:
        inches = 12

    cmap = plt.cm.jet
    normalized = inches / 12.0
    rgba = cmap(normalized)
    return np.array([rgba[0], rgba[1], rgba[2], 0.5])


def create_slope_colorbar():
    """Create a discrete legend for slope percentages."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ranges_labels = []

    positive_color = value_to_rgb(0.01)
    ranges_labels.append(("> 0%", positive_color))

    negative_ranges = [
        ("0% to -0.5%", -0.0025),
        ("-0.5% to -1.0%", -0.0075),
        ("-1.0% to -1.5%", -0.0125),
        ("-1.5% to -2.0%", -0.0175),
        ("-2.0% to -2.5%", -0.0225),
        ("-2.5% to -3.0%", -0.0275),
        ("-3.0% to -3.5%", -0.0325),
        ("< -3.5%", -0.035)
    ]

    for label, sample_value in negative_ranges:
        color = value_to_rgb(sample_value)
        ranges_labels.append((label, color))

    patches = [mpatches.Patch(color=color[:3], alpha=color[3], label=label) for label, color in ranges_labels]

    legend = ax.legend(handles=patches, loc='center', title='Slope Percentages',
                       title_fontsize=12, fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def create_displacement_colorbar():
    """Create a discrete legend for displacement values (0 to 12 inches)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ranges_labels = []

    displacement_ranges = [
        ("\u2264 0\"", 0),
        ("2\"", 2.0),
        ("4\"", 4.0),
        ("6\"", 6.0),
        ("8\"", 8.0),
        ("10\"", 10.0),
        ("\u2265 12\"", 12.0)
    ]

    for label, sample_value in displacement_ranges:
        color = displacement_to_rgb(sample_value)
        ranges_labels.append((label, color))

    patches = [mpatches.Patch(color=color[:3], alpha=color[3], label=label) for label, color in ranges_labels]

    legend = ax.legend(handles=patches, loc='center', title='Displacement (in)',
                       title_fontsize=12, fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Slope Percentages Legend:")
    create_slope_colorbar()

    print("Displacement Legend:")
    create_displacement_colorbar()
