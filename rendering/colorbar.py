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
try:
    from config import SLOPE_RANGE_POSITIVE, SLOPE_RANGE_NEGATIVE
except ImportError:
    SLOPE_RANGE_POSITIVE = None
    SLOPE_RANGE_NEGATIVE = None


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


def _slope_gradient(ax, range_pos, range_neg):
    """Render an asymmetric jet gradient for slope onto *ax*.

    x-axis runs from +range_pos (blue, left) to -range_neg (red, right),
    representing deviation from expected slope in percent.
    Zero (on design) is positioned proportionally.
    """
    n = 512
    values = np.linspace(range_pos, -range_neg, n)
    mapped = np.where(
        values >= 0,
        0.5 - (values / range_pos) * 0.5,
        0.5 + (-values / range_neg) * 0.5,
    )
    colors = plt.cm.jet(mapped)
    ax.imshow(colors.reshape(1, -1, 4), aspect='auto',
              extent=[range_pos, -range_neg, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(range_pos, -range_neg)


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
    _save_or_show(fig, save_path)


def create_new_slope_colorbar(save_path=None):
    """Standalone new slope colorbar — same style as combined legend."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_new_slope_bar(ax)
    _save_or_show(fig, save_path)



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
    """Draw asymmetric slope colorbar onto an existing axes."""
    expected_pct = EXPECTED_WALL_SLOPE * 100
    range_pos = SLOPE_RANGE_POSITIVE if SLOPE_RANGE_POSITIVE is not None else SLOPE_COLORMAP_RANGE
    range_neg = SLOPE_RANGE_NEGATIVE if SLOPE_RANGE_NEGATIVE is not None else SLOPE_COLORMAP_RANGE

    _slope_gradient(ax, range_pos, range_neg)

    # Ticks in deviation space, labels show actual slope %
    ticks = [range_pos, range_pos / 2, 0, -range_neg / 2, -range_neg]
    labels = [
        f"{expected_pct + range_pos:.1f}%",
        f"{expected_pct + range_pos / 2:.1f}%",
        f"{expected_pct:.1f}%",
        f"{expected_pct - range_neg / 2:.1f}%",
        f"{expected_pct - range_neg:.1f}%",
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
    zero_frac = range_pos / (range_pos + range_neg)
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
    ax.set_xlabel(f"Slope (expected: {expected_pct:.1f}%)",
                  fontsize=22, fontweight='bold', labelpad=55)


def _draw_new_slope_bar(ax):
    """Draw asymmetric slope deviation colorbar onto an existing axes."""
    expected_pct = EXPECTED_WALL_SLOPE * 100
    range_pos = SLOPE_RANGE_POSITIVE if SLOPE_RANGE_POSITIVE is not None else SLOPE_COLORMAP_RANGE
    range_neg = SLOPE_RANGE_NEGATIVE if SLOPE_RANGE_NEGATIVE is not None else SLOPE_COLORMAP_RANGE

    _slope_gradient(ax, range_pos, range_neg)

    # Ticks in deviation space
    ticks = [range_pos, range_pos / 2, 0, -range_neg / 2, -range_neg]
    labels = [
        f"+{range_pos:.1f}%",
        f"+{range_pos / 2:.1f}%",
        "0%",
        f"-{range_neg / 2:.1f}%",
        f"-{range_neg:.1f}%",
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
    zero_frac = range_pos / (range_pos + range_neg)
    positions = [(0.05, 'Top further back', 'blue'),
                 (zero_frac, 'On design', 'green'),
                 (0.95, 'Top further forward', 'red')]
    for i, (x, text, color) in enumerate(positions):
        y = 1.1
        for j in range(i):
            if abs(x - positions[j][0]) < 0.18:
                y += 0.20
        ax.annotate(text, xy=(x, y), xycoords='axes fraction',
                    ha='center', fontsize=21, color=color, fontweight='bold')
    ax.set_xlabel(f"Top-of-Wall Deviation from Expected ({expected_pct:.1f}%)",
                  fontsize=22, fontweight='bold', labelpad=55)


def _settlement_gradient(ax, max_in):
    """Render a coolwarm gradient from 0 (blue) to max (red) for settlement."""
    n = 512
    values = np.linspace(0, 1, n)
    colors = plt.cm.coolwarm(values)
    ax.imshow(colors.reshape(1, -1, 4), aspect='auto',
              extent=[0, max_in, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(0, max_in)


def _rotation_gradient(ax, range_pct):
    """Render a symmetric coolwarm gradient for rotation (dZ/dX)."""
    n = 512
    values = np.linspace(0, 1, n)
    colors = plt.cm.coolwarm(values)
    ax.imshow(colors.reshape(1, -1, 4), aspect='auto',
              extent=[-range_pct, range_pct, 0, 1])
    ax.set_yticks([])
    ax.set_xlim(-range_pct, range_pct)


def _draw_settlement_bar(ax, max_settle_m):
    """Draw settlement colorbar: blue=0 (no settlement) to red=max settlement.

    max_settle_m: actual 95th percentile settlement in meters (from analysis).
    """
    max_in = max_settle_m * METERS_TO_FEET * 12

    _settlement_gradient(ax, max_in)

    ticks = [0, max_in / 4, max_in / 2, 3 * max_in / 4, max_in]
    labels = [
        "0\"",
        f"{max_in / 4:.3f}\"",
        f"{max_in / 2:.3f}\"",
        f"{3 * max_in / 4:.3f}\"",
        f"{max_in:.3f}\"",
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
    positions = [(0.05, 'No settlement', 'blue'),
                 (0.95, 'Max settlement', 'red')]
    for i, (x, text, color) in enumerate(positions):
        ax.annotate(text, xy=(x, 1.1), xycoords='axes fraction',
                    ha='center', fontsize=21, color=color, fontweight='bold')
    ax.set_xlabel("Settlement (relative to highest point on joint)",
                  fontsize=22, fontweight='bold', labelpad=55)


def _draw_rotation_bar(ax, max_rotation):
    """Draw rotation colorbar: symmetric coolwarm around 0.

    max_rotation: actual 95th percentile |dZ/dX| from analysis (dimensionless).
    """
    range_pct = max_rotation * 100  # convert to percent

    _rotation_gradient(ax, range_pct)

    ticks = [-range_pct, -range_pct / 2, 0, range_pct / 2, range_pct]
    labels = [
        f"-{range_pct:.2f}%",
        f"-{range_pct / 2:.2f}%",
        "0%",
        f"+{range_pct / 2:.2f}%",
        f"+{range_pct:.2f}%",
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
    positions = [(0.05, 'Tilting left', 'blue'),
                 (0.5, 'Level', 'grey'),
                 (0.95, 'Tilting right', 'red')]
    for i, (x, text, color) in enumerate(positions):
        ax.annotate(text, xy=(x, 1.1), xycoords='axes fraction',
                    ha='center', fontsize=21, color=color, fontweight='bold')
    ax.set_xlabel("Joint Rotation (dZ/dX slope along joint)",
                  fontsize=22, fontweight='bold', labelpad=55)


def create_settlement_colorbar(max_settle_m, save_path=None):
    """Standalone settlement colorbar. max_settle_m is the 95th pctile in meters."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_settlement_bar(ax, max_settle_m)
    _save_or_show(fig, save_path)


def create_rotation_colorbar(max_rotation, save_path=None):
    """Standalone rotation colorbar. max_rotation is the 95th pctile |dZ/dX|."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_rotation_bar(ax, max_rotation)
    _save_or_show(fig, save_path)


def _draw_vdisp_bar(ax, max_disp_m):
    """Draw vertical displacement colorbar: symmetric coolwarm.

    max_disp_m: 95th percentile |X(z) - X(base)| in meters.
    """
    max_in = max_disp_m * METERS_TO_FEET * 12

    _rotation_gradient(ax, max_in)  # symmetric gradient works here

    ticks = [-max_in, -max_in / 2, 0, max_in / 2, max_in]
    labels = [
        f'-{max_in:.3f}"',
        f'-{max_in / 2:.3f}"',
        '0"',
        f'+{max_in / 2:.3f}"',
        f'+{max_in:.3f}"',
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
    positions = [(0.05, 'Shifted left', 'blue'),
                 (0.5, 'On base', 'grey'),
                 (0.95, 'Shifted right', 'red')]
    for i, (x, text, color) in enumerate(positions):
        ax.annotate(text, xy=(x, 1.1), xycoords='axes fraction',
                    ha='center', fontsize=21, color=color, fontweight='bold')
    ax.set_xlabel("Vertical Joint Displacement (X deviation from base)",
                  fontsize=22, fontweight='bold', labelpad=55)


def _draw_vrot_bar(ax, max_rotation):
    """Draw vertical rotation colorbar: symmetric coolwarm.

    max_rotation: 95th percentile |dX/dZ| (dimensionless).
    """
    range_pct = max_rotation * 100

    _rotation_gradient(ax, range_pct)

    ticks = [-range_pct, -range_pct / 2, 0, range_pct / 2, range_pct]
    labels = [
        f"-{range_pct:.2f}%",
        f"-{range_pct / 2:.2f}%",
        "0%",
        f"+{range_pct / 2:.2f}%",
        f"+{range_pct:.2f}%",
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
    positions = [(0.05, 'Leaning left', 'blue'),
                 (0.5, 'Plumb', 'grey'),
                 (0.95, 'Leaning right', 'red')]
    for i, (x, text, color) in enumerate(positions):
        ax.annotate(text, xy=(x, 1.1), xycoords='axes fraction',
                    ha='center', fontsize=21, color=color, fontweight='bold')
    ax.set_xlabel("Vertical Joint Rotation (dX/dZ along joint)",
                  fontsize=22, fontweight='bold', labelpad=55)


def create_vdisp_colorbar(max_disp_m, save_path=None):
    """Standalone vertical displacement colorbar."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_vdisp_bar(ax, max_disp_m)
    _save_or_show(fig, save_path)


def create_vrot_colorbar(max_rotation, save_path=None):
    """Standalone vertical rotation colorbar."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _draw_vrot_bar(ax, max_rotation)
    _save_or_show(fig, save_path)


def create_all_colorbars(save_dir="outputs/images/"):
    """Generate individual and combined legend images.

    Settlement and rotation legends are NOT generated here — they require
    actual data ranges computed by analysis/joint_detection.py at runtime.
    """
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


if __name__ == "__main__":
    create_all_colorbars()
