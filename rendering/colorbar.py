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


def _staggered_ticks(ax, ticks, labels, fontsize=14, min_gap_pts=80,
                     row_height=22, base_tick=8):
    """Draw tick labels below *ax*, bumping labels down when they'd overlap.

    For each label left-to-right, check if its centre is within
    *min_gap_pts* points of any label already placed at the same row.
    If so, try the next row down, repeating until clear.

    Returns the deepest y-offset used (in points below the axes) so the
    caller can place a title underneath.
    """
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)

    # Convert data coords to display (points) for distance checks
    fig = ax.get_figure()
    fig.canvas.draw()  # needed so transforms are current
    display_xs = []
    for t in ticks:
        disp = ax.transData.transform((t, 0))
        display_xs.append(disp[0])  # x in display pixels

    # Assign rows by checking overlap with already-placed labels
    rows = [0] * len(ticks)
    for i in range(len(ticks)):
        for attempt_row in range(10):
            conflict = False
            for j in range(i):
                if rows[j] == attempt_row:
                    dist = abs(display_xs[i] - display_xs[j])
                    if dist < min_gap_pts:
                        conflict = True
                        break
            if not conflict:
                rows[i] = attempt_row
                break

    max_row = max(rows)
    for i, (t, label) in enumerate(zip(ticks, labels)):
        row = rows[i]
        tick_len = base_tick + row * row_height
        label_y = -(tick_len + 4)
        ax.annotate('', xy=(t, 0), xytext=(t, -tick_len),
                    xycoords=('data', 'axes points'),
                    textcoords=('data', 'axes points'),
                    arrowprops=dict(arrowstyle='-', color='k', lw=1))
        ax.annotate(label, xy=(t, label_y),
                    xycoords=('data', 'axes points'),
                    ha='center', va='top', fontsize=fontsize)

    return base_tick + max_row * row_height + fontsize + 10


def _save_or_show(fig, save_path):
    """Save figure to file or display it."""
    if save_path:
        ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def create_displacement_colorbar(save_path=None):
    """Standalone displacement colorbar — same style as combined legend."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_displacement_bar(ax)
    _save_or_show(fig, save_path)


def create_slope_colorbar(save_path=None):
    """Standalone slope colorbar — same style as combined legend."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_slope_bar(ax)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def create_new_slope_colorbar(save_path=None):
    """Standalone new slope colorbar — same style as combined legend."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_new_slope_bar(ax)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def create_all_colorbars(save_dir="outputs/images/"):
    """Generate individual and combined legend images."""
    os.makedirs(save_dir, exist_ok=True)

    create_displacement_colorbar(save_path=os.path.join(save_dir, "legend_displacement.png"))
    create_slope_colorbar(save_path=os.path.join(save_dir, "legend_slope.png"))
    create_new_slope_colorbar(save_path=os.path.join(save_dir, "legend_new_slope.png"))

    combined_path = os.path.join(save_dir, "legend_colorbars.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    for ax_idx, create_fn in enumerate([
        _draw_displacement_bar,
        _draw_slope_bar,
        _draw_new_slope_bar,
    ]):
        create_fn(axes[ax_idx])

    plt.tight_layout(h_pad=3.0)
    fig.savefig(combined_path, dpi=150, bbox_inches='tight', transparent=True)
    print(f"Saved: {combined_path}")
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
        "0\"",
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
                    ha='center', va='top', fontsize=18)
    ax.tick_params(axis='x', length=0)
    zero_frac = max_pos_in / (max_pos_in + max_neg_in)
    # Stagger annotations vertically when they'd overlap
    positions = [(0.05, 'More batter', 'blue'),
                 (zero_frac, 'On design', 'green'),
                 (0.95, 'Less batter', 'red')]
    for i, (x, text, color) in enumerate(positions):
        y = 1.1
        for j in range(i):
            if abs(x - positions[j][0]) < 0.18:
                y += 0.20
        ax.annotate(text, xy=(x, y), xycoords='axes fraction',
                    ha='center', fontsize=21, color=color, fontweight='bold')
    ax.set_xlabel(f"Displacement from Expected Profile (batter: {expected_pct:.1f}%)",
                  fontsize=22, fontweight='bold', labelpad=55)


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
        f"{expected_pct:.1f}%",
        f"{expected_pct-r/2:.1f}%",
        f"{expected_pct-r:.1f}%",
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    ax.tick_params(axis='x', labelsize=18)
    ax.annotate('More batter', xy=(0.05, 1.1), xycoords='axes fraction',
                ha='center', fontsize=21, color='blue', fontweight='bold')
    ax.annotate('On design', xy=(0.5, 1.1), xycoords='axes fraction',
                ha='center', fontsize=21, color='green', fontweight='bold')
    ax.annotate('Less batter', xy=(0.95, 1.1), xycoords='axes fraction',
                ha='center', fontsize=21, color='red', fontweight='bold')
    ax.set_xlabel(f"Slope (expected: {expected_pct:.1f}%, range: \u00b1{r:.1f}%)",
                  fontsize=22, fontweight='bold', labelpad=15)


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
        "0%",
        f"-{r/2:.1f}%",
        f"-{r:.1f}% dev",
    ]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    ax.tick_params(axis='x', labelsize=18)
    ax.annotate('Top further back', xy=(0.05, 1.1), xycoords='axes fraction',
                ha='center', fontsize=21, color='blue', fontweight='bold')
    ax.annotate('On design', xy=(0.5, 1.1), xycoords='axes fraction',
                ha='center', fontsize=21, color='green', fontweight='bold')
    ax.annotate('Top further forward', xy=(0.95, 1.1), xycoords='axes fraction',
                ha='center', fontsize=21, color='red', fontweight='bold')
    ax.set_xlabel(f"Top-of-Wall Deviation from Expected ({expected_pct:.1f}%)",
                  fontsize=22, fontweight='bold', labelpad=15)


if __name__ == "__main__":
    create_all_colorbars()
