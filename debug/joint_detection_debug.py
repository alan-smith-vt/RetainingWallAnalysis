"""
Debug visualization for joint detection pipeline.

Produces diagnostic plots:
  1. 2D joint score raster (horizontal and vertical)
  2. Sample vertical profiles with detected peaks
  3. All tracked joints overlaid on the raster
  4. Normals cross-section with tracked joints

Run from repo root: python debug/joint_detection_debug.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import glob

from analysis.joint_detection import (
    compute_normals, rasterize_joint_score, detect_joints_windowed,
    track_joints, filter_tracks, fit_track_splines,
    BLOCK_HEIGHT_M, BLOCK_WIDTH_M,
)
from config import (
    JOINT_NORMAL_KNN, JOINT_RASTER_RESOLUTION,
    JOINT_GAUSSIAN_SIGMA, JOINT_PEAK_MIN_HEIGHT,
    JOINT_WINDOW_WIDTH, JOINT_WINDOW_STEP,
    JOINT_MATCH_TOLERANCE, JOINT_MIN_TRACK_LENGTH,
    JOINT_SPLINE_SMOOTHING, JOINT_SPLINE_POINTS,
    BLOCK_HEIGHT_IN, BLOCK_WIDTH_IN,
)

try:
    from config import CURVE_DEBUG_CENTER_X, CURVE_DEBUG_RANGE
except ImportError:
    CURVE_DEBUG_CENTER_X = None
    CURVE_DEBUG_RANGE = 2.0

OUTPUT_DIR = "outputs/debug"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_rasters(raster_hz, raster_vt, x_edges, z_edges, output_path):
    """Plot 1: Joint score rasters."""
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_centers[0], x_centers[-1], z_centers[0], z_centers[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax0 = axes[0]
    im0 = ax0.imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
                     cmap='hot', interpolation='nearest')
    plt.colorbar(im0, ax=ax0, label='|Nz| score', shrink=0.8)
    ax0.set_ylabel("Z (m)")
    ax0.set_title("Horizontal Joint Score (|Nz|) — block height = %d in" % BLOCK_HEIGHT_IN)

    ax1 = axes[1]
    im1 = ax1.imshow(raster_vt, aspect='auto', origin='lower', extent=extent,
                     cmap='hot', interpolation='nearest')
    plt.colorbar(im1, ax=ax1, label='|Nx| score', shrink=0.8)
    ax1.set_xlabel("X — along wall (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title("Vertical Joint Score (|Nx|) — block width = %d in" % BLOCK_WIDTH_IN)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_vertical_profiles(raster_hz, x_edges, z_edges, detections,
                           output_path, n_samples=6):
    """Plot 2: Sample vertical profiles with peaks marked."""
    if not detections:
        print("  No detections, skipping vertical profiles plot.")
        return

    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    resolution = JOINT_RASTER_RESOLUTION

    # Pick evenly spaced sample windows
    indices = np.linspace(0, len(detections) - 1, min(n_samples, len(detections)),
                          dtype=int)

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 6),
                             sharey=True)
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        center_x, peak_z, peak_scores = detections[idx]

        # Reconstruct the averaged column for this window
        col_idx = np.argmin(np.abs(x_centers - center_x))
        window_cols = max(1, int(JOINT_WINDOW_WIDTH / resolution))
        col_start = max(0, col_idx - window_cols // 2)
        col_end = min(raster_hz.shape[1], col_start + window_cols)
        column = np.mean(raster_hz[:, col_start:col_end], axis=1)

        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(column, sigma=JOINT_GAUSSIAN_SIGMA)

        ax.plot(column, z_centers, 'gray', alpha=0.4, linewidth=0.8, label='Raw')
        ax.plot(smoothed, z_centers, 'steelblue', linewidth=1.5, label='Smoothed')
        ax.scatter(peak_scores, peak_z, color='red', s=50, zorder=5, label='Peaks')

        # Draw expected block height grid from first peak
        if len(peak_z) > 0:
            z0 = peak_z[0]
            z = z0
            while z <= z_centers[-1]:
                ax.axhline(z, color='orange', alpha=0.3, linewidth=0.8, linestyle='--')
                z += BLOCK_HEIGHT_M
            z = z0 - BLOCK_HEIGHT_M
            while z >= z_centers[0]:
                ax.axhline(z, color='orange', alpha=0.3, linewidth=0.8, linestyle='--')
                z -= BLOCK_HEIGHT_M

        ax.set_title(f"X={center_x:.1f}m", fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Z (m)", fontsize=8)
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle("Vertical Profiles with Detected Peaks\n"
                 "(orange dashes = expected block height grid)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_tracked_joints(raster_hz, x_edges, z_edges, tracks, splines,
                        output_path):
    """Plot 3: Raster with tracked joints and spline fits overlaid.

    Joints shown in green/blue, cracks shown in red/orange.
    """
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_centers[0], x_centers[-1], z_centers[0], z_centers[-1]]

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
              cmap='hot', interpolation='nearest', alpha=0.6)

    joint_tracks = [t for t in tracks if t.get('label') == 'joint']
    crack_tracks = [t for t in tracks if t.get('label') == 'crack']

    # Plot joints with distinct colors + splines
    joint_colors = plt.cm.winter(np.linspace(0, 1, max(len(joint_tracks), 1)))
    spline_idx = 0
    for i, track in enumerate(joint_tracks):
        c = joint_colors[i % len(joint_colors)]
        ax.scatter(track['x'], track['z'], color=c, s=15, zorder=4,
                   edgecolors='white', linewidths=0.3)
        if spline_idx < len(splines) and splines[spline_idx]['tck'] is not None:
            ax.plot(splines[spline_idx]['x'], splines[spline_idx]['z'],
                    color=c, linewidth=2, zorder=5)
        spline_idx += 1

    # Plot cracks in red
    for track in crack_tracks:
        ax.scatter(track['x'], track['z'], color='red', s=15, zorder=4,
                   edgecolors='white', linewidths=0.3, marker='x')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='teal',
               markersize=8, label='Joint (%d)' % len(joint_tracks)),
        Line2D([0], [0], marker='x', color='red', linewidth=0,
               markersize=8, label='Crack (%d)' % len(crack_tracks)),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

    ax.set_xlabel("X — along wall (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Tracked Horizontal Joints (%d joints, %d cracks, block=%din x %din)" % (
        len(joint_tracks), len(crack_tracks), BLOCK_HEIGHT_IN, BLOCK_WIDTH_IN))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        print("No displacement PLY files found.")
        return

    ensure_dir(OUTPUT_DIR)

    for file in files:
        wall_id = os.path.basename(file).split("_")[1]
        print(f"\n=== Joint Detection Debug: Wall {wall_id} ===")

        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.positions.numpy()

        # Optionally restrict to debug region
        center_x = CURVE_DEBUG_CENTER_X
        x_range = CURVE_DEBUG_RANGE
        if center_x is not None:
            mask = ((points[:, 0] >= center_x - x_range) &
                    (points[:, 0] <= center_x + x_range))
            points = points[mask]
            print(f"  Debug region: X = {center_x:.1f} ± {x_range:.1f}m")

        print(f"  {len(points)} points")

        print("  Computing normals...")
        normals = compute_normals(points, JOINT_NORMAL_KNN)

        print("  Rasterizing...")
        resolution = JOINT_RASTER_RESOLUTION
        raster_hz, raster_vt, x_edges, z_edges = rasterize_joint_score(
            points, normals, resolution)
        print(f"  Raster shape: {raster_hz.shape}")

        print("  Detecting peaks...")
        detections = detect_joints_windowed(
            raster_hz, x_edges, z_edges,
            window_width=JOINT_WINDOW_WIDTH,
            window_step=JOINT_WINDOW_STEP,
            sigma=JOINT_GAUSSIAN_SIGMA,
            min_height=JOINT_PEAK_MIN_HEIGHT,
            resolution=resolution,
        )
        print(f"  {len(detections)} windows, "
              f"{sum(len(d[1]) for d in detections)} total peaks")

        print("  Tracking...")
        tracks = track_joints(detections, JOINT_MATCH_TOLERANCE)
        tracks = filter_tracks(tracks, JOINT_MIN_TRACK_LENGTH)
        print(f"  {len(tracks)} tracks")

        print("  Fitting splines...")
        splines = fit_track_splines(tracks, JOINT_SPLINE_SMOOTHING, JOINT_SPLINE_POINTS)

        prefix = f"{OUTPUT_DIR}/wall_{wall_id}_joints"

        plot_rasters(raster_hz, raster_vt, x_edges, z_edges,
                     f"{prefix}_raster.png")
        plot_vertical_profiles(raster_hz, x_edges, z_edges, detections,
                               f"{prefix}_profiles.png")
        plot_tracked_joints(raster_hz, x_edges, z_edges, tracks, splines,
                            f"{prefix}_tracked.png")

    print(f"\nDone. Debug plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
