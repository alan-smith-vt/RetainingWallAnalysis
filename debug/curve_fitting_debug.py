"""
Debug visualization for settlement curve fitting.

Extracts a small X range from the unrolled displacement point cloud and
produces diagnostic plots showing:
  1. Per-slice vertical density histograms with detected peaks
  2. Point cloud cross-section colored by density with peaks overlaid
  3. Raw peak scatter and local spline fit

Usage:
    python debug/curve_fitting_debug.py

Configure the debug region via CURVE_DEBUG_* settings in config.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
import glob

from config import (
    CURVE_WINDOW_THICKNESS, CURVE_STEP_SIZE, CURVE_BIN_WIDTH,
    CURVE_CROP_DELTA, CURVE_OUTLIER_THRESHOLD,
    CURVE_SMOOTHING_FACTOR_1, CURVE_SMOOTHING_FACTOR_2,
    CURVE_NUM_SMOOTH_POINTS, DENSITY_RADIUS,
)

# ── Debug settings (override in config.py if desired) ────────────────────────
try:
    from config import CURVE_DEBUG_CENTER_X, CURVE_DEBUG_RANGE
except ImportError:
    CURVE_DEBUG_CENTER_X = None  # None = use midpoint of point cloud
    CURVE_DEBUG_RANGE = 2.0      # ±meters around center

OUTPUT_DIR = "outputs/debug"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_debug_region(points, center_x, x_range):
    """Extract points within [center_x - x_range, center_x + x_range]."""
    mask = (points[:, 0] >= center_x - x_range) & (points[:, 0] <= center_x + x_range)
    return points[mask]


def compute_density_histogram(slice_points, bin_width, slice_axis=2):
    """Compute vertical density histogram for a single slice.

    Returns (bin_centers, bin_counts, peak_z, peak_count) or None if empty.
    """
    if len(slice_points) < 2:
        return None

    sorted_indices = np.argsort(slice_points[:, slice_axis])
    sorted_points_raw = slice_points[sorted_indices]
    sorted_points = sorted_points_raw - sorted_points_raw[0, slice_axis]

    z_min = sorted_points[0, slice_axis]
    z_max = sorted_points[-1, slice_axis]
    z_range = z_max - z_min
    num_bins = int(z_range // bin_width)

    if num_bins < 2:
        return None

    z_values = sorted_points[:, slice_axis]

    counts = []
    prev_idx = 0
    for i in range(1, num_bins):
        target_z = bin_width * i
        next_idx = np.searchsorted(z_values, target_z)
        counts.append(next_idx - prev_idx)
        prev_idx = next_idx

    counts = np.array(counts)
    bin_edges = np.linspace(0, 1, num_bins) * z_range - z_min
    bin_centers = bin_edges[1:] - bin_width / 2

    # Map back to original Z coordinates
    original_z_min = slice_points[sorted_indices[0], slice_axis]
    bin_centers_abs = bin_centers + original_z_min

    max_idx = np.argmax(counts)
    peak_z = bin_centers_abs[max_idx]
    peak_count = counts[max_idx]

    return bin_centers_abs, counts, peak_z, peak_count


def sort_and_slice(points, window_thickness, step_size, axis=0):
    """Sliding window along axis. Returns list of (window_center_x, slice_points)."""
    sorted_indices = np.argsort(points[:, axis])
    sorted_points = points[sorted_indices]

    x_min = sorted_points[0, axis]
    x_max = sorted_points[-1, axis]
    x_values = sorted_points[:, axis]

    slices = []
    window_start = x_min

    while window_start <= x_max:
        window_end = window_start + window_thickness
        start_idx = np.searchsorted(x_values, window_start, side='left')
        end_idx = np.searchsorted(x_values, window_end, side='left')

        if start_idx < end_idx:
            center_x = (window_start + window_end) / 2
            slices.append((center_x, sorted_points[start_idx:end_idx]))

        window_start += step_size
        if window_start > x_max:
            break

    return slices


def plot_density_histograms(slices, bin_width, output_path):
    """Plot 1: Per-slice vertical density histograms in a grid."""
    n = len(slices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (center_x, slice_pts) in enumerate(slices):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        result = compute_density_histogram(slice_pts, bin_width)
        if result is None:
            ax.set_title(f"X={center_x:.1f}m\n(no data)")
            ax.set_visible(False)
            continue

        bin_centers, counts, peak_z, peak_count = result
        ax.barh(bin_centers, counts, height=bin_width * 0.9, color='steelblue', alpha=0.7)
        ax.axhline(peak_z, color='red', linewidth=1.5, linestyle='--', label=f'Peak Z={peak_z:.3f}')
        ax.scatter(peak_count, peak_z, color='red', s=60, zorder=5)
        ax.set_title(f"X={center_x:.1f}m", fontsize=9)
        ax.set_xlabel("Count", fontsize=8)
        ax.set_ylabel("Z (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Vertical Density Histograms per Slice", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_cross_section(region_points, peaks, output_path):
    """Plot 2: Point cloud cross-section colored by local density, peaks overlaid."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute point density for coloring
    if len(region_points) > 0:
        tree = KDTree(region_points[:, [0, 2]])
        counts = tree.query_ball_point(region_points[:, [0, 2]], r=DENSITY_RADIUS, return_length=True)
        norm = Normalize(vmin=np.min(counts), vmax=np.max(counts))

        sc = ax.scatter(region_points[:, 0], region_points[:, 2],
                        c=counts, cmap='viridis', norm=norm,
                        s=1, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label='Neighbor count', shrink=0.8)

    if len(peaks) > 0:
        ax.scatter(peaks[:, 0], peaks[:, 1], color='red', s=40, zorder=5,
                   edgecolors='white', linewidths=0.5, label='Density peaks')
        ax.legend(fontsize=9)

    ax.set_xlabel("X — along wall (m)")
    ax.set_ylabel("Z — elevation (m)")
    ax.set_title("Point Cloud Cross-Section with Detected Peaks")
    ax.set_aspect('auto')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_peak_scatter_and_spline(peaks, output_path):
    """Plot 3: Raw peaks, outlier removal, and spline fit."""
    if len(peaks) < 4:
        print("  Not enough peaks for spline fit, skipping plot 3.")
        return

    x = peaks[:, 0]
    z = peaks[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: raw peaks + first-pass spline
    ax1 = axes[0]
    ax1.scatter(x, z, color='magenta', s=20, label='Raw peaks', zorder=3)

    try:
        tck, u = splprep([x, z], s=CURVE_SMOOTHING_FACTOR_1)
        u_dense = np.linspace(0, 1, 500)
        x_fit, z_fit = splev(u_dense, tck)
        ax1.plot(x_fit, z_fit, 'b-', linewidth=1.5, label=f'Spline pass 1 (s={CURVE_SMOOTHING_FACTOR_1})')

        # Mark outliers
        u_vals = x / np.max(x)
        _, z_smooth = splev(u_vals, tck)
        residuals = np.abs(z - z_smooth)
        outlier_mask = residuals > CURVE_OUTLIER_THRESHOLD
        if np.any(outlier_mask):
            ax1.scatter(x[outlier_mask], z[outlier_mask], color='red', s=60, marker='x',
                        linewidths=2, label=f'Outliers (>{CURVE_OUTLIER_THRESHOLD}m)', zorder=4)
    except Exception as e:
        ax1.text(0.5, 0.5, f"Spline fit failed: {e}", transform=ax1.transAxes, ha='center')

    ax1.set_ylabel("Z — elevation (m)")
    ax1.set_title("Pass 1: Raw Peaks + Spline + Outlier Detection")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: cleaned peaks + second-pass spline
    ax2 = axes[1]
    try:
        x_clean = x[~outlier_mask]
        z_clean = z[~outlier_mask]
        ax2.scatter(x_clean, z_clean, color='magenta', s=20, label='Cleaned peaks', zorder=3)

        tck2, u2 = splprep([x_clean, z_clean], s=CURVE_SMOOTHING_FACTOR_2)
        x_fit2, z_fit2 = splev(u_dense, tck2)
        ax2.plot(x_fit2, z_fit2, 'b-', linewidth=1.5, label=f'Spline pass 2 (s={CURVE_SMOOTHING_FACTOR_2})')
    except Exception as e:
        ax2.scatter(x, z, color='magenta', s=20, label='Peaks (no outlier removal)')
        ax2.text(0.5, 0.5, f"Second pass failed: {e}", transform=ax2.transAxes, ha='center')

    ax2.set_xlabel("X — along wall (m)")
    ax2.set_ylabel("Z — elevation (m)")
    ax2.set_title("Pass 2: Cleaned Peaks + Final Spline")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Peak Detection and Spline Fitting", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        print("No displacement PLY files found in outputs/point_clouds/unrolled/")
        return

    ensure_dir(OUTPUT_DIR)

    for file in files:
        wall_id = os.path.basename(file).split("_")[1]
        print(f"\n=== Debug: Wall {wall_id} — {file} ===")

        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.positions.numpy()

        # Determine debug center
        center_x = CURVE_DEBUG_CENTER_X
        if center_x is None:
            center_x = (np.min(points[:, 0]) + np.max(points[:, 0])) / 2
        x_range = CURVE_DEBUG_RANGE

        print(f"  Debug region: X = {center_x:.1f} ± {x_range:.1f}m")
        print(f"  Point cloud X range: [{np.min(points[:, 0]):.1f}, {np.max(points[:, 0]):.1f}]")

        region = extract_debug_region(points, center_x, x_range)
        print(f"  Points in region: {len(region)}")

        if len(region) == 0:
            print("  No points in debug region, skipping.")
            continue

        # Run sliding window and peak detection on the debug region
        slices = sort_and_slice(region, CURVE_WINDOW_THICKNESS, CURVE_STEP_SIZE, axis=0)
        print(f"  Slices in region: {len(slices)}")

        # Collect peaks
        peaks = []
        for center, slice_pts in slices:
            result = compute_density_histogram(slice_pts, CURVE_BIN_WIDTH)
            if result is not None:
                _, _, peak_z, _ = result
                peaks.append([center, peak_z])
        peaks = np.array(peaks) if peaks else np.empty((0, 2))
        print(f"  Peaks detected: {len(peaks)}")

        prefix = f"{OUTPUT_DIR}/wall_{wall_id}"

        # Plot 1: Density histograms per slice
        plot_density_histograms(slices, CURVE_BIN_WIDTH, f"{prefix}_histograms.png")

        # Plot 2: Cross-section with peaks
        plot_cross_section(region, peaks, f"{prefix}_cross_section.png")

        # Plot 3: Peak scatter and spline fit
        plot_peak_scatter_and_spline(peaks, f"{prefix}_spline_fit.png")

    print(f"\nDone. Debug plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
