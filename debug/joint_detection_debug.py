"""
Debug visualization for joint detection pipeline.

Operates on a small region of the wall (CURVE_DEBUG_CENTER_X ± CURVE_DEBUG_RANGE).
All pipeline functions are inline — no dependency on analysis/joint_detection.py.

Produces diagnostic plots:
  1. 2D joint score raster (horizontal and vertical)
  2. Sample vertical profiles with detected peaks
  3. All tracked joints overlaid on the raster

Run from repo root: python debug/joint_detection_debug.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import glob

from config import FEET_TO_METERS

# ── Settings (import from config if present, otherwise use defaults) ─────────

def _cfg(name, default):
    try:
        from config import __dict__ as _c  # noqa
    except Exception:
        pass
    try:
        mod = sys.modules.get('config') or __import__('config')
        return getattr(mod, name, default)
    except Exception:
        return default

CURVE_DEBUG_CENTER_X  = _cfg('CURVE_DEBUG_CENTER_X', None)
CURVE_DEBUG_RANGE     = _cfg('CURVE_DEBUG_RANGE', 2.0)

NORMAL_KNN            = _cfg('JOINT_NORMAL_KNN', 30)
RASTER_RESOLUTION     = _cfg('JOINT_RASTER_RESOLUTION', 0.01)
GAUSSIAN_SIGMA        = _cfg('JOINT_GAUSSIAN_SIGMA', 3)
PEAK_MIN_HEIGHT       = _cfg('JOINT_PEAK_MIN_HEIGHT', 0.15)
WINDOW_WIDTH          = _cfg('JOINT_WINDOW_WIDTH', 1.0)
WINDOW_STEP           = _cfg('JOINT_WINDOW_STEP', 0.5)
MATCH_TOLERANCE       = _cfg('JOINT_MATCH_TOLERANCE', 0.05)
MIN_TRACK_LENGTH      = _cfg('JOINT_MIN_TRACK_LENGTH', 5)
SPLINE_SMOOTHING      = _cfg('JOINT_SPLINE_SMOOTHING', 1.0)
SPLINE_POINTS         = _cfg('JOINT_SPLINE_POINTS', 500)
BLOCK_HEIGHT_IN       = _cfg('BLOCK_HEIGHT_IN', 8)
BLOCK_WIDTH_IN        = _cfg('BLOCK_WIDTH_IN', 16)
BLOCK_HEIGHT_TOLERANCE = _cfg('BLOCK_HEIGHT_TOLERANCE', 0.3)

BLOCK_HEIGHT_M = BLOCK_HEIGHT_IN * FEET_TO_METERS / 12
BLOCK_WIDTH_M  = BLOCK_WIDTH_IN * FEET_TO_METERS / 12

OUTPUT_DIR = "outputs/debug"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ── Pipeline functions (self-contained) ──────────────────────────────────────

def compute_normals(points, knn):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals)


def rasterize_joint_score(points, normals, resolution, axis_h=0, axis_v=2):
    x = points[:, axis_h]
    z = points[:, axis_v]

    x_edges = np.arange(x.min(), x.max() + resolution, resolution)
    z_edges = np.arange(z.min(), z.max() + resolution, resolution)

    xi = np.clip(np.searchsorted(x_edges, x) - 1, 0, len(x_edges) - 2)
    zi = np.clip(np.searchsorted(z_edges, z) - 1, 0, len(z_edges) - 2)

    n_cols = len(x_edges) - 1
    n_rows = len(z_edges) - 1
    flat_idx = zi * n_cols + xi

    nz_abs = np.abs(normals[:, axis_v])
    nx_abs = np.abs(normals[:, axis_h])

    hz_sum = np.bincount(flat_idx, weights=nz_abs, minlength=n_rows * n_cols)
    vt_sum = np.bincount(flat_idx, weights=nx_abs, minlength=n_rows * n_cols)
    counts = np.bincount(flat_idx, minlength=n_rows * n_cols)

    mask = counts > 0
    raster_hz = np.zeros(n_rows * n_cols)
    raster_vt = np.zeros(n_rows * n_cols)
    raster_hz[mask] = hz_sum[mask] / counts[mask]
    raster_vt[mask] = vt_sum[mask] / counts[mask]

    return raster_hz.reshape(n_rows, n_cols), raster_vt.reshape(n_rows, n_cols), x_edges, z_edges


def detect_peaks_in_column(column, z_centers, sigma, min_height,
                           min_distance_bins, expected_spacing_bins=None,
                           spacing_tolerance=None):
    if len(column) < 3:
        return np.array([]), np.array([]), column

    smoothed = gaussian_filter1d(column, sigma=sigma)

    if expected_spacing_bins is not None and spacing_tolerance is not None:
        min_dist = max(min_distance_bins,
                       int(expected_spacing_bins * (1 - spacing_tolerance)))
    else:
        min_dist = min_distance_bins

    peak_indices, _ = find_peaks(smoothed, height=min_height, distance=min_dist)

    if len(peak_indices) == 0:
        return np.array([]), np.array([]), smoothed

    return z_centers[peak_indices], smoothed[peak_indices], smoothed


def detect_joints_windowed(raster_hz, x_edges, z_edges, resolution):
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    expected_spacing_bins = max(1, int(BLOCK_HEIGHT_M / resolution))
    min_distance_bins = max(1, int(BLOCK_HEIGHT_M * 0.5 / resolution))
    window_cols = max(1, int(WINDOW_WIDTH / resolution))
    step_cols = max(1, int(WINDOW_STEP / resolution))

    detections = []
    col = 0
    while col < raster_hz.shape[1]:
        col_end = min(col + window_cols, raster_hz.shape[1])
        window_avg = np.mean(raster_hz[:, col:col_end], axis=1)
        center_x = np.mean(x_centers[col:col_end])

        peak_z, peak_scores, _ = detect_peaks_in_column(
            window_avg, z_centers, GAUSSIAN_SIGMA, PEAK_MIN_HEIGHT,
            min_distance_bins,
            expected_spacing_bins=expected_spacing_bins,
            spacing_tolerance=BLOCK_HEIGHT_TOLERANCE,
        )

        if len(peak_z) > 0:
            detections.append((center_x, peak_z, peak_scores))

        col += step_cols

    return detections


def track_joints(detections):
    tracks = []
    active_track_z = []

    for x, peak_z, peak_scores in detections:
        used_tracks = set()
        used_peaks = set()

        if active_track_z:
            track_z_arr = np.array(active_track_z)
            dists = np.abs(track_z_arr[:, None] - peak_z[None, :])

            for flat_idx in np.argsort(dists.ravel()):
                ti, pi = divmod(int(flat_idx), len(peak_z))
                if ti in used_tracks or pi in used_peaks:
                    continue
                if dists[ti, pi] > MATCH_TOLERANCE:
                    break
                tracks[ti]['x'].append(x)
                tracks[ti]['z'].append(peak_z[pi])
                tracks[ti]['score'].append(peak_scores[pi])
                active_track_z[ti] = peak_z[pi]
                used_tracks.add(ti)
                used_peaks.add(pi)

        for pi in range(len(peak_z)):
            if pi not in used_peaks:
                tracks.append({'x': [x], 'z': [peak_z[pi]], 'score': [peak_scores[pi]]})
                active_track_z.append(peak_z[pi])

    return tracks


def filter_tracks(tracks):
    return [t for t in tracks if len(t['x']) >= MIN_TRACK_LENGTH]


def classify_tracks(tracks):
    if not tracks:
        return []

    for t in tracks:
        t['mean_z'] = np.mean(t['z'])
    tracks_sorted = sorted(tracks, key=lambda t: t['mean_z'])

    expected = BLOCK_HEIGHT_M
    tolerance = BLOCK_HEIGHT_TOLERANCE

    def spacing_ok(z1, z2):
        gap = abs(z2 - z1)
        if expected <= 0:
            return True
        n = round(gap / expected)
        return n > 0 and abs(gap - n * expected) <= expected * tolerance

    n = len(tracks_sorted)
    for i, t in enumerate(tracks_sorted):
        has_neighbor = False
        if i > 0 and spacing_ok(tracks_sorted[i - 1]['mean_z'], t['mean_z']):
            has_neighbor = True
        if i < n - 1 and spacing_ok(t['mean_z'], tracks_sorted[i + 1]['mean_z']):
            has_neighbor = True
        t['label'] = 'joint' if has_neighbor else 'crack'

    return tracks_sorted


def fit_track_splines(tracks):
    splines = []
    for track in tracks:
        x = np.array(track['x'])
        z = np.array(track['z'])
        if len(x) < 4:
            splines.append({'x': x, 'z': z, 'tck': None})
            continue
        try:
            tck, u = splprep([x, z], s=SPLINE_SMOOTHING)
            u_new = np.linspace(0, 1, SPLINE_POINTS)
            x_s, z_s = splev(u_new, tck)
            splines.append({'x': x_s, 'z': z_s, 'tck': tck})
        except Exception:
            splines.append({'x': x, 'z': z, 'tck': None})
    return splines


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_rasters(raster_hz, raster_vt, x_edges, z_edges, output_path):
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    im0 = axes[0].imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
                         cmap='hot', interpolation='nearest')
    plt.colorbar(im0, ax=axes[0], label='|Nz| score', shrink=0.8)
    axes[0].set_ylabel("Z (m)")
    axes[0].set_title("Horizontal Joint Score (|Nz|) — block height = %d in" % BLOCK_HEIGHT_IN)

    im1 = axes[1].imshow(raster_vt, aspect='auto', origin='lower', extent=extent,
                         cmap='hot', interpolation='nearest')
    plt.colorbar(im1, ax=axes[1], label='|Nx| score', shrink=0.8)
    axes[1].set_xlabel("X — along wall (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].set_title("Vertical Joint Score (|Nx|) — block width = %d in" % BLOCK_WIDTH_IN)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_vertical_profiles(raster_hz, x_edges, z_edges, detections,
                           output_path, n_samples=6):
    if not detections:
        print("  No detections, skipping vertical profiles plot.")
        return

    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2

    indices = np.linspace(0, len(detections) - 1,
                          min(n_samples, len(detections)), dtype=int)

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 6),
                             sharey=True)
    if len(indices) == 1:
        axes = [axes]

    window_cols = max(1, int(WINDOW_WIDTH / RASTER_RESOLUTION))

    for ax, idx in zip(axes, indices):
        center_x, peak_z, peak_scores = detections[idx]

        col_idx = np.argmin(np.abs(x_centers - center_x))
        col_start = max(0, col_idx - window_cols // 2)
        col_end = min(raster_hz.shape[1], col_start + window_cols)
        column = np.mean(raster_hz[:, col_start:col_end], axis=1)
        smoothed = gaussian_filter1d(column, sigma=GAUSSIAN_SIGMA)

        ax.plot(column, z_centers, 'gray', alpha=0.4, linewidth=0.8, label='Raw')
        ax.plot(smoothed, z_centers, 'steelblue', linewidth=1.5, label='Smoothed')
        ax.scatter(peak_scores, peak_z, color='red', s=50, zorder=5, label='Peaks')

        # Expected block height grid from first peak
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
    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2
    extent = [x_c[0], x_c[-1], z_c[0], z_c[-1]]

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.imshow(raster_hz, aspect='auto', origin='lower', extent=extent,
              cmap='hot', interpolation='nearest', alpha=0.6)

    joint_tracks = [t for t in tracks if t.get('label') == 'joint']
    crack_tracks = [t for t in tracks if t.get('label') == 'crack']

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

    for track in crack_tracks:
        ax.scatter(track['x'], track['z'], color='red', s=15, zorder=4,
                   edgecolors='white', linewidths=0.3, marker='x')

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
    ax.set_title("Tracked Joints (%d joints, %d cracks, block=%din x %din)" % (
        len(joint_tracks), len(crack_tracks), BLOCK_HEIGHT_IN, BLOCK_WIDTH_IN))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        print("No displacement PLY files found in outputs/point_clouds/unrolled/")
        return

    ensure_dir(OUTPUT_DIR)

    for file in files:
        wall_id = os.path.basename(file).split("_")[1]
        print(f"\n=== Joint Detection Debug: Wall {wall_id} ===")

        pc = o3d.t.io.read_point_cloud(file)
        points = pc.point.positions.numpy()
        print(f"  Full point cloud: {len(points)} points, "
              f"X=[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]")

        # Always crop to debug region
        center_x = CURVE_DEBUG_CENTER_X
        if center_x is None:
            center_x = (points[:, 0].min() + points[:, 0].max()) / 2
        x_range = CURVE_DEBUG_RANGE

        mask = ((points[:, 0] >= center_x - x_range) &
                (points[:, 0] <= center_x + x_range))
        points = points[mask]
        print(f"  Debug region: X = {center_x:.1f} +/- {x_range:.1f}m -> {len(points)} points")

        if len(points) == 0:
            print("  No points in debug region, skipping.")
            continue

        print("  Computing normals...")
        normals = compute_normals(points, NORMAL_KNN)

        print("  Rasterizing...")
        raster_hz, raster_vt, x_edges, z_edges = rasterize_joint_score(
            points, normals, RASTER_RESOLUTION)
        print(f"  Raster shape: {raster_hz.shape}")

        print("  Detecting peaks...")
        detections = detect_joints_windowed(raster_hz, x_edges, z_edges,
                                           RASTER_RESOLUTION)
        print(f"  {len(detections)} windows, "
              f"{sum(len(d[1]) for d in detections)} total peaks")

        print("  Tracking and classifying...")
        tracks = track_joints(detections)
        tracks = filter_tracks(tracks)
        tracks = classify_tracks(tracks)

        joint_tracks = [t for t in tracks if t['label'] == 'joint']
        crack_tracks = [t for t in tracks if t['label'] == 'crack']
        print(f"  {len(joint_tracks)} joints, {len(crack_tracks)} cracks")

        print("  Fitting splines...")
        splines = fit_track_splines(joint_tracks)

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
