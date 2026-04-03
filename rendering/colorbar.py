"""
Colorbar / legend generator.

Creates reference colorbar images that match the centered jet colormaps
used in the displacement and slope analyses. Reads all parameters from
config.py so the legends always stay in sync with the analysis output.

Outputs:
  outputs/images/legend_displacement_colorbar.png
  outputs/images/legend_slope_colorbar.png
  outputs/images/legend_new_slope_colorbar.png
  outputs/images/legend_all_colorbars.png  (combined)
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
try:
    from config import MAX_DISPLACEMENT_POSITIVE, MAX_DISPLACEMENT_NEGATIVE
except ImportError:
    MAX_DISPLACEMENT_POSITIVE = None
    MAX_DISPLACEMENT_NEGATIVE = None


def ensure_dir(filepath):
    """Create parent directories for a file path if they don't exist."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def _displacement_gradient(ax, max_pos_in, max_neg_in):
    """Render an asymmetric jet gradient into *ax* with real-unit x-axis.

    The x-axis runs from +max_pos_in (blue, left) to -max_neg_in (red, right).
    Zero is positioned proportionally, so it is off-center when the ranges differ.
    """
    n = 512
    values = np.linspace(max_pos_in, -max_neg_in, n)
    # Same mapping as rendering/point_cloud.py scalar_to_colors_displacement
    mapped = np.where(
        values >= 0,
        0.5 - (values / max_pos_in) * 0.5,
        0.5 + (-values / max_neg_in) * 0.5,
    )
    colors = plt.cm.jet(mapped)
    ax.imshow(colors.reshape(1, -1, 4), aspect='auto',
              extent=[max_pos_in, -max_neg_in, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(max_pos_in, -max_neg_in)


def create_displacement_colorbar(save_path=None):
    """
    Asymmetric jet colorbar for displacement analysis.

    Blue = more batter than expected (wall behind profile)
    Green/Yellow = on expected profile
    Red = less batter than expected (wall forward of profile)
    """
    fig, ax = plt.subplots(figsize=(5, 3))

    max_pos_m = MAX_DISPLACEMENT_POSITIVE if MAX_DISPLACEMENT_POSITIVE is not None else MAX_DISPLACEMENT_FOR_COLORS
    max_neg_m = MAX_DISPLACEMENT_NEGATIVE if MAX_DISPLACEMENT_NEGATIVE is not None else MAX_DISPLACEMENT_FOR_COLORS
    max_pos_in = max_pos_m * METERS_TO_FEET * 12
    max_neg_in = max_neg_m * METERS_TO_FEET * 12

    _displacement_gradient(ax, max_pos_in, max_neg_in)

    # Place tick labels manually at staggered heights so they don't overlap
    ticks = [max_pos_in, max_pos_in / 2, 0, -max_neg_in / 2, -max_neg_in]
    labels = [
        f"+{max_pos_in:.1f}\"",
        f"+{max_pos_in / 2:.1f}\"",
        "0 (on profile)",
        f"-{max_neg_in / 2:.1f}\"",
        f"-{max_neg_in:.1f}\"",
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels([])  # hide default labels
    # Draw long ticks manually + staggered labels
    for i, (t, label) in enumerate(zip(ticks, labels)):
        is_low = (i % 2 == 1)  # stagger odd-indexed labels lower
        tick_len = 18 if is_low else 10
        label_y = -tick_len - 18
        ax.plot([t, t], [0, 0], color='k', clip_on=False,
                transform=ax.get_xaxis_transform())
        # Draw tick line below the bar
        ax.annotate('', xy=(t, 0), xytext=(t, -tick_len),
                    xycoords=('data', 'axes points'),
                    textcoords=('data', 'axes points'),
                    arrowprops=dict(arrowstyle='-', color='k', lw=1))
        ax.annotate(label, xy=(t, label_y),
                    xycoords=('data', 'axes points'),
                    ha='center', va='top', fontsize=14)
    ax.tick_params(axis='x', length=0)  # hide default ticks

    expected_pct = EXPECTED_WALL_SLOPE * 100
    title = (f"Displacement from Expected Profile "
             f"(expected batter: {expected_pct:.1f}%)")
    fig.text(0.5, 0.02, title, ha='center', fontsize=17, fontweight='bold')

    # Zero fraction for annotation positioning
    zero_frac = max_pos_in / (max_pos_in + max_neg_in)
    ax.annotate('More batter\nthan expected', xy=(0.05, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='blue', fontweight='bold')
    ax.annotate('On profile', xy=(zero_frac, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='green', fontweight='bold')
    ax.annotate('Less batter\nthan expected', xy=(0.95, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='red', fontweight='bold')

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
    fig, ax = plt.subplots(figsize=(5, 3))

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

    expected_pct = EXPECTED_WALL_SLOPE * 100
    range_pct = SLOPE_COLORMAP_RANGE

    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = [
        f"{expected_pct + range_pct:.1f}%",
        f"{expected_pct + range_pct/2:.1f}%",
        f"{expected_pct:.1f}% (expected)",
        f"{expected_pct - range_pct/2:.1f}%",
        f"{expected_pct - range_pct:.1f}%",
    ]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    ax.tick_params(axis='x', labelsize=14)

    title = (f"Piecewise Slope  |  "
             f"expected: {expected_pct:.1f}%  |  "
             f"range: \u00b1{range_pct:.1f}%")
    fig.text(0.5, 0.02, title, ha='center', fontsize=17, fontweight='bold')

    ax.annotate('More batter', xy=(0.05, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='blue', fontweight='bold')
    ax.annotate('On design', xy=(0.5, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='green', fontweight='bold')
    ax.annotate('Less batter', xy=(0.95, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='red', fontweight='bold')

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
    fig, ax = plt.subplots(figsize=(5, 3))

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')

    range_pct = SLOPE_COLORMAP_RANGE
    expected_pct = EXPECTED_WALL_SLOPE * 100

    tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = [
        f"+{range_pct:.1f}% deviation",
        f"+{range_pct/2:.1f}%",
        f"0% (matches {expected_pct:.1f}%)",
        f"-{range_pct/2:.1f}%",
        f"-{range_pct:.1f}% deviation",
    ]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)
    ax.tick_params(axis='x', labelsize=14)

    title = (f"Top-of-Wall Slope Deviation from Expected ({expected_pct:.1f}%)  |  "
             f"range: \u00b1{range_pct:.1f}%")
    fig.text(0.5, 0.02, title, ha='center', fontsize=17, fontweight='bold')

    ax.annotate('Top further back', xy=(0.05, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='blue', fontweight='bold')
    ax.annotate('On design', xy=(0.5, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='green', fontweight='bold')
    ax.annotate('Top further forward', xy=(0.95, 1.15), xycoords='axes fraction',
                ha='center', fontsize=12, color='red', fontweight='bold')

    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def create_all_colorbars(save_dir="outputs/images/"):
    """Generate all three colorbars and a combined image."""
    os.makedirs(save_dir, exist_ok=True)

    disp_path = os.path.join(save_dir, "legend_displacement_colorbar.png")
    slope_path = os.path.join(save_dir, "legend_slope_colorbar.png")
    new_slope_path = os.path.join(save_dir, "legend_new_slope_colorbar.png")
    combined_path = os.path.join(save_dir, "legend_all_colorbars.png")

    create_displacement_colorbar(save_path=disp_path)
    create_slope_colorbar(save_path=slope_path)
    create_new_slope_colorbar(save_path=new_slope_path)

    # Stack all three into one image
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

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
    """Draw asymmetric displacement colorbar onto an existing axes."""
    max_pos_m = MAX_DISPLACEMENT_POSITIVE if MAX_DISPLACEMENT_POSITIVE is not None else MAX_DISPLACEMENT_FOR_COLORS
    max_neg_m = MAX_DISPLACEMENT_NEGATIVE if MAX_DISPLACEMENT_NEGATIVE is not None else MAX_DISPLACEMENT_FOR_COLORS
    max_pos_in = max_pos_m * METERS_TO_FEET * 12
    max_neg_in = max_neg_m * METERS_TO_FEET * 12
    expected_pct = EXPECTED_WALL_SLOPE * 100

    _displacement_gradient(ax, max_pos_in, max_neg_in)

    ticks = [max_pos_in, max_pos_in / 2, 0, -max_neg_in / 2, -max_neg_in]
    labels = [
        f"+{max_pos_in:.1f}\"",
        f"+{max_pos_in / 2:.1f}\"",
        "0 (on profile)",
        f"-{max_neg_in / 2:.1f}\"",
        f"-{max_neg_in:.1f}\"",
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    for i, (t, label) in enumerate(zip(ticks, labels)):
        is_low = (i % 2 == 1)
        tick_len = 14 if is_low else 8
        label_y = -tick_len - 14
        ax.annotate('', xy=(t, 0), xytext=(t, -tick_len),
                    xycoords=('data', 'axes points'),
                    textcoords=('data', 'axes points'),
                    arrowprops=dict(arrowstyle='-', color='k', lw=1))
        ax.annotate(label, xy=(t, label_y),
                    xycoords=('data', 'axes points'),
                    ha='center', va='top', fontsize=12)
    ax.tick_params(axis='x', length=0)
    ax.set_xlabel(f"Displacement from Expected Profile (batter: {expected_pct:.1f}%)",
                  fontsize=15, fontweight='bold', labelpad=50)


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
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel(f"Piecewise Slope (expected: {expected_pct:.1f}%, range: \u00b1{r:.1f}%)",
                  fontsize=15, fontweight='bold', labelpad=15)


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
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel(f"Top-of-Wall Deviation from Expected ({expected_pct:.1f}%)",
                  fontsize=15, fontweight='bold', labelpad=15)


if __name__ == "__main__":
    create_all_colorbars()
