"""
Colorbar / legend generator.

Creates reference colorbar images that match the centered jet colormaps
used in the displacement and slope analyses. Reads all parameters from
config.py so the legends always stay in sync with the analysis output.

Outputs:
  renders/legends/displacement_colorbar.png
  renders/legends/slope_colorbar.png
  renders/legends/new_slope_colorbar.png
  renders/legends/all_colorbars.png  (combined)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from config import (
    MAX_DISPLACEMENT_FOR_COLORS, EXPECTED_WALL_SLOPE,
    SLOPE_COLORMAP_RANGE, METERS_TO_FEET,
)


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def create_displacement_colorbar(save_path=None):
    """
    Centered jet colorbar for displacement analysis.

    Blue = more batter than expected (wall behind profile)
    Green/Yellow = on expected profile
    Red = less batter than expected (wall forward of profile)

    Range: -MAX_DISPLACEMENT_FOR_COLORS to +MAX_DISPLACEMENT_FOR_COLORS
    """
    fig, ax = plt.subplots(figsize=(10, 2))

    max_disp_m = MAX_DISPLACEMENT_FOR_COLORS
    max_disp_ft = max_disp_m * METERS_TO_FEET
    max_disp_in = max_disp_ft * 12

    # Build the colorbar from the same mapping used in wall_analysis.py:
    #   mapped = clip(delta / MAX * 0.5 + 0.5, 0, 1)
    # So delta = -MAX → 0 (blue), delta = 0 → 0.5 (green), delta = +MAX → 1 (red)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

    # Label tick positions in real units
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels_m = [
        f"-{max_disp_m:.2f} m\n(-{max_disp_in:.1f}\")",
        f"-{max_disp_m/2:.2f} m\n(-{max_disp_in/2:.1f}\")",
        f"0 m\n(on profile)",
        f"+{max_disp_m/2:.2f} m\n(+{max_disp_in/2:.1f}\")",
        f"+{max_disp_m:.2f} m\n(+{max_disp_in:.1f}\")",
    ]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels_m)
    ax.tick_params(labelsize=9)

    expected_pct = EXPECTED_WALL_SLOPE * 100
    title = (f"Displacement from Expected Profile "
             f"(expected batter: {expected_pct:.1f}%)")
    cbar.set_label(title, fontsize=11, fontweight='bold')

    # Annotations
    ax.annotate('More batter\nthan expected', xy=(0.05, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='blue', fontweight='bold')
    ax.annotate('On profile', xy=(0.5, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='green', fontweight='bold')
    ax.annotate('Less batter\nthan expected', xy=(0.95, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def create_slope_colorbar(save_path=None):
    """
    Centered jet colorbar for piecewise slope analysis.

    Blue = measured slope > expected (more batter)
    Green/Yellow = measured slope matches expected
    Red = measured slope < expected (less batter)

    Range: expected - SLOPE_COLORMAP_RANGE% to expected + SLOPE_COLORMAP_RANGE%
    """
    fig, ax = plt.subplots(figsize=(10, 2))

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

    expected_pct = EXPECTED_WALL_SLOPE * 100
    range_pct = SLOPE_COLORMAP_RANGE

    # The mapping in wall_analysis.py value_to_rgb_jet:
    #   deviation = measured - expected
    #   mapped = clip((-deviation * 100) / RANGE * 0.5 + 0.5, 0, 1)
    # So deviation = +RANGE% → 0 (blue), deviation = 0 → 0.5 (green), deviation = -RANGE% → 1 (red)
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = [
        f"{expected_pct + range_pct:.1f}%",
        f"{expected_pct + range_pct/2:.1f}%",
        f"{expected_pct:.1f}%\n(expected)",
        f"{expected_pct - range_pct/2:.1f}%",
        f"{expected_pct - range_pct:.1f}%",
    ]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    ax.tick_params(labelsize=9)

    title = (f"Piecewise Slope  |  "
             f"expected: {expected_pct:.1f}%  |  "
             f"range: \u00b1{range_pct:.1f}%")
    cbar.set_label(title, fontsize=11, fontweight='bold')

    ax.annotate('More batter', xy=(0.05, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='blue', fontweight='bold')
    ax.annotate('On design', xy=(0.5, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='green', fontweight='bold')
    ax.annotate('Less batter', xy=(0.95, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def create_new_slope_colorbar(save_path=None):
    """
    Centered jet colorbar for the new slope (top-vs-bottom deviation) analysis.

    Same color convention but values represent deviation from expected.
    Blue = top of wall further back than expected
    Red = top of wall further forward than expected
    """
    fig, ax = plt.subplots(figsize=(10, 2))

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

    range_pct = SLOPE_COLORMAP_RANGE
    expected_pct = EXPECTED_WALL_SLOPE * 100

    # new_slope is already a deviation; mapping:
    #   mapped = clip((-new_slope * 100) / RANGE * 0.5 + 0.5, 0, 1)
    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = [
        f"+{range_pct:.1f}%\ndeviation",
        f"+{range_pct/2:.1f}%",
        f"0%\n(matches {expected_pct:.1f}%)",
        f"-{range_pct/2:.1f}%",
        f"-{range_pct:.1f}%\ndeviation",
    ]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    ax.tick_params(labelsize=9)

    title = (f"Top-of-Wall Slope Deviation from Expected ({expected_pct:.1f}%)  |  "
             f"range: \u00b1{range_pct:.1f}%")
    cbar.set_label(title, fontsize=11, fontweight='bold')

    ax.annotate('Top further back', xy=(0.05, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='blue', fontweight='bold')
    ax.annotate('On design', xy=(0.5, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='green', fontweight='bold')
    ax.annotate('Top further forward', xy=(0.95, 1.15), xycoords='axes fraction',
                ha='center', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def create_all_colorbars(save_dir="renders/legends/"):
    """Generate all three colorbars and a combined image."""
    os.makedirs(save_dir, exist_ok=True)

    disp_path = os.path.join(save_dir, "displacement_colorbar.png")
    slope_path = os.path.join(save_dir, "slope_colorbar.png")
    new_slope_path = os.path.join(save_dir, "new_slope_colorbar.png")
    combined_path = os.path.join(save_dir, "all_colorbars.png")

    create_displacement_colorbar(save_path=disp_path)
    create_slope_colorbar(save_path=slope_path)
    create_new_slope_colorbar(save_path=new_slope_path)

    # Stack all three into one image
    fig, axes = plt.subplots(3, 1, figsize=(10, 7))

    for ax_idx, (create_fn, title) in enumerate([
        (_draw_displacement_bar, "Displacement"),
        (_draw_slope_bar, "Piecewise Slope"),
        (_draw_new_slope_bar, "Top-of-Wall Deviation"),
    ]):
        create_fn(axes[ax_idx])

    plt.tight_layout(h_pad=3.0)
    fig.savefig(combined_path, dpi=150, bbox_inches='tight', transparent=True)
    print(f"Saved combined: {combined_path}")
    plt.close(fig)


def _draw_displacement_bar(ax):
    """Draw displacement colorbar onto an existing axes."""
    max_disp_m = MAX_DISPLACEMENT_FOR_COLORS
    max_disp_in = max_disp_m * METERS_TO_FEET * 12
    expected_pct = EXPECTED_WALL_SLOPE * 100

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')

    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    labels = [
        f"-{max_disp_m:.2f}m ({-max_disp_in:.1f}\")",
        f"-{max_disp_m/2:.2f}m",
        "0 (on profile)",
        f"+{max_disp_m/2:.2f}m",
        f"+{max_disp_m:.2f}m (+{max_disp_in:.1f}\")",
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    ax.tick_params(labelsize=8)
    cbar.set_label(f"Displacement from Expected Profile (batter: {expected_pct:.1f}%)",
                   fontsize=10, fontweight='bold')


def _draw_slope_bar(ax):
    """Draw slope colorbar onto an existing axes."""
    expected_pct = EXPECTED_WALL_SLOPE * 100
    r = SLOPE_COLORMAP_RANGE

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')

    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    labels = [
        f"{expected_pct+r:.1f}%",
        f"{expected_pct+r/2:.1f}%",
        f"{expected_pct:.1f}% (expected)",
        f"{expected_pct-r/2:.1f}%",
        f"{expected_pct-r:.1f}%",
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    ax.tick_params(labelsize=8)
    cbar.set_label(f"Piecewise Slope (expected: {expected_pct:.1f}%, range: \u00b1{r:.1f}%)",
                   fontsize=10, fontweight='bold')


def _draw_new_slope_bar(ax):
    """Draw new slope deviation colorbar onto an existing axes."""
    expected_pct = EXPECTED_WALL_SLOPE * 100
    r = SLOPE_COLORMAP_RANGE

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')

    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    labels = [
        f"+{r:.1f}% dev",
        f"+{r/2:.1f}%",
        f"0% (= {expected_pct:.1f}%)",
        f"-{r/2:.1f}%",
        f"-{r:.1f}% dev",
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    ax.tick_params(labelsize=8)
    cbar.set_label(f"Top-of-Wall Deviation from Expected ({expected_pct:.1f}%)",
                   fontsize=10, fontweight='bold')


if __name__ == "__main__":
    create_all_colorbars()
