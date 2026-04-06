"""
Interactive joint detection parameter tuner.

Computes normals once on startup, then serves a web UI with sliders
for all tuning parameters. Each slider change re-runs rasterization,
peak detection, tracking, and plot generation on the server.

Run from repo root:
    python debug/joint_tuner.py

Then open http://localhost:5000 in your browser.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from flask import Flask, request, jsonify, send_from_directory
import base64
from io import BytesIO
import glob
import json

from config import FEET_TO_METERS

# ── Load config defaults ────────────────────────────────────────────────────

def _cfg(name, default):
    try:
        mod = sys.modules.get('config') or __import__('config')
        return getattr(mod, name, default)
    except Exception:
        return default

DEFAULTS = {
    'raster_resolution': _cfg('JOINT_RASTER_RESOLUTION', 0.01),
    'gaussian_sigma':    _cfg('JOINT_GAUSSIAN_SIGMA', 3.0),
    'peak_min_height':   _cfg('JOINT_PEAK_MIN_HEIGHT', 0.15),
    'peak_min_dist_m':   0.05,
    'window_width':      _cfg('JOINT_WINDOW_WIDTH', 1.0),
    'window_step':       _cfg('JOINT_WINDOW_STEP', 0.5),
    'match_tolerance':   _cfg('JOINT_MATCH_TOLERANCE', 0.05),
    'min_track_length':  _cfg('JOINT_MIN_TRACK_LENGTH', 5),
    'spline_smoothing':  _cfg('JOINT_SPLINE_SMOOTHING', 1.0),
    'block_height_in':   _cfg('BLOCK_HEIGHT_IN', 8),
    'block_height_tol':  _cfg('BLOCK_HEIGHT_TOLERANCE', 0.3),
}

NORMAL_KNN = _cfg('JOINT_NORMAL_KNN', 30)
DEBUG_CENTER = _cfg('CURVE_DEBUG_CENTER_X', None)
DEBUG_RANGE = _cfg('CURVE_DEBUG_RANGE', 2.0)

# ── Pipeline functions ──────────────────────────────────────────────────────

def rasterize(points, normals, resolution):
    x, z = points[:, 0], points[:, 2]
    nx = normals[:, 0]
    x_edges = np.arange(x.min(), x.max() + resolution, resolution)
    z_edges = np.arange(z.min(), z.max() + resolution, resolution)

    xi = np.clip(np.searchsorted(x_edges, x) - 1, 0, len(x_edges) - 2)
    zi = np.clip(np.searchsorted(z_edges, z) - 1, 0, len(z_edges) - 2)

    n_cols, n_rows = len(x_edges) - 1, len(z_edges) - 1
    flat = zi * n_cols + xi
    total = n_rows * n_cols

    hz_sum = np.bincount(flat, weights=np.abs(normals[:, 2]), minlength=total)
    counts = np.bincount(flat, minlength=total)

    # Separate Nx+ (right-facing) and Nx- (left-facing)
    nx_pos = np.clip(nx, 0, None)   # only positive Nx
    nx_neg = np.clip(-nx, 0, None)  # only negative Nx (flipped to positive)
    vt_pos_sum = np.bincount(flat, weights=nx_pos, minlength=total)
    vt_neg_sum = np.bincount(flat, weights=nx_neg, minlength=total)

    m = counts > 0
    raster_hz = np.zeros(total)
    raster_vt_pos = np.zeros(total)
    raster_vt_neg = np.zeros(total)
    raster_hz[m] = hz_sum[m] / counts[m]
    raster_vt_pos[m] = vt_pos_sum[m] / counts[m]
    raster_vt_neg[m] = vt_neg_sum[m] / counts[m]

    return (raster_hz.reshape(n_rows, n_cols),
            raster_vt_pos.reshape(n_rows, n_cols),
            raster_vt_neg.reshape(n_rows, n_cols),
            x_edges, z_edges)


def detect_peaks_horizontal(raster, x_edges, z_edges, params):
    """Detect horizontal joints: slide windows along X, find peaks in Z profiles."""
    res = params['raster_resolution']
    sigma = params['gaussian_sigma']
    min_h = params['peak_min_height']
    min_dist_m = params.get('peak_min_dist_m', 0.05)

    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2

    min_dist_bins = max(1, int(min_dist_m / res))
    win_cols = max(1, int(params['window_width'] / res))
    step_cols = max(1, int(params['window_step'] / res))

    detections = []
    col = 0
    while col < raster.shape[1]:
        ce = min(col + win_cols, raster.shape[1])
        avg = np.mean(raster[:, col:ce], axis=1)
        cx = np.mean(x_c[col:ce])

        if len(avg) < 3:
            col += step_cols
            continue

        sm = gaussian_filter1d(avg, sigma=sigma)
        pi, _ = find_peaks(sm, height=min_h, distance=min_dist_bins)

        if len(pi) > 0:
            detections.append((cx, z_c[pi], sm[pi]))
        col += step_cols

    return detections


def detect_peaks_vertical(raster, x_edges, z_edges, params):
    """Detect vertical joints: slide windows along Z, find peaks in X profiles."""
    res = params['raster_resolution']
    sigma = params['gaussian_sigma']
    min_h = params['peak_min_height']
    min_dist_m = params.get('peak_min_dist_m', 0.05)

    x_c = (x_edges[:-1] + x_edges[1:]) / 2
    z_c = (z_edges[:-1] + z_edges[1:]) / 2

    min_dist_bins = max(1, int(min_dist_m / res))
    win_rows = max(1, int(params['window_width'] / res))
    step_rows = max(1, int(params['window_step'] / res))

    detections = []
    row = 0
    while row < raster.shape[0]:
        re_ = min(row + win_rows, raster.shape[0])
        avg = np.mean(raster[row:re_, :], axis=0)
        cz = np.mean(z_c[row:re_])

        if len(avg) < 3:
            row += step_rows
            continue

        sm = gaussian_filter1d(avg, sigma=sigma)
        pi, _ = find_peaks(sm, height=min_h, distance=min_dist_bins)

        if len(pi) > 0:
            # For vertical: detection = (window_center_z, peak_x_values, peak_scores)
            detections.append((cz, x_c[pi], sm[pi]))
        row += step_rows

    return detections


def track_adjacent(detections, params):
    """Link peaks between adjacent windows only.

    Works for both horizontal and vertical modes — the first element of each
    detection is the window center (x for horizontal, z for vertical) and the
    second is the array of peak positions in the other axis.
    """
    tol = params['match_tolerance']
    min_len = int(params['min_track_length'])

    if not detections:
        return []

    finished = []
    wc0, pp0, ps0 = detections[0]
    # Each track stores 'w' (window-axis coords) and 'p' (peak-axis coords)
    active = [{'w': [wc0], 'p': [pp0[i]], 's': [ps0[i]]} for i in range(len(pp0))]

    for det_idx in range(1, len(detections)):
        wc, pp, ps = detections[det_idx]
        new_active = []
        used_p = set()

        if active and len(pp) > 0:
            active_pos = np.array([t['p'][-1] for t in active])
            dists = np.abs(active_pos[:, None] - pp[None, :])

            used_t = set()
            for fi in np.argsort(dists.ravel()):
                ti, pi = divmod(int(fi), len(pp))
                if ti in used_t or pi in used_p:
                    continue
                if dists[ti, pi] > tol:
                    break
                active[ti]['w'].append(wc)
                active[ti]['p'].append(pp[pi])
                active[ti]['s'].append(ps[pi])
                new_active.append(active[ti])
                used_t.add(ti)
                used_p.add(pi)

            for ti in range(len(active)):
                if ti not in used_t:
                    finished.append(active[ti])
        else:
            finished.extend(active)

        for pi in range(len(pp)):
            if pi not in used_p:
                new_active.append({'w': [wc], 'p': [pp[pi]], 's': [ps[pi]]})

        active = new_active

    finished.extend(active)

    # Convert to x/z format and filter by min length
    tracks = []
    for t in finished:
        if len(t['w']) >= min_len:
            tracks.append(t)
    return tracks


def tracks_to_xz(tracks, mode):
    """Convert generic w/p tracks to x/z tracks for plotting.

    Horizontal: w=x, p=z.  Vertical (pos/neg): w=z, p=x.
    """
    result = []
    for t in tracks:
        if mode == 'horizontal':
            result.append({'x': t['w'], 'z': t['p'], 's': t['s'],
                           'mz': np.mean(t['p'])})
        else:  # vertical_pos or vertical_neg
            result.append({'x': t['p'], 'z': t['w'], 's': t['s'],
                           'mz': np.mean(t['w'])})
    return result


def classify_tracks(tracks, params):
    """Classify tracks as joint or crack based on block spacing."""
    block_h = params['block_height_in'] * FEET_TO_METERS / 12
    bt = params['block_height_tol']

    tracks.sort(key=lambda t: t['mz'])

    def ok(z1, z2):
        g = abs(z2 - z1)
        n = round(g / block_h) if block_h > 0 else 0
        return n > 0 and abs(g - n * block_h) <= block_h * bt

    for i, t in enumerate(tracks):
        nb = False
        if i > 0 and ok(tracks[i-1]['mz'], t['mz']):
            nb = True
        if i < len(tracks)-1 and ok(t['mz'], tracks[i+1]['mz']):
            nb = True
        t['label'] = 'joint' if nb else 'crack'

    return tracks


def fit_splines(tracks, params):
    sm = params['spline_smoothing']
    splines = []
    for t in tracks:
        x, z = np.array(t['x']), np.array(t['z'])
        if len(x) < 4:
            splines.append({'x': x, 'z': z, 'ok': False})
            continue
        try:
            tck, _ = splprep([x, z], s=sm)
            u = np.linspace(0, 1, 500)
            xs, zs = splev(u, tck)
            splines.append({'x': xs, 'z': zs, 'ok': True})
        except Exception:
            splines.append({'x': x, 'z': z, 'ok': False})
    return splines


# ── Plot generation ─────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_raster_plot(raster_hz, raster_vt_pos, raster_vt_neg, x_edges, z_edges, params, mode):
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    zc = (z_edges[:-1] + z_edges[1:]) / 2
    ext = [xc[0], xc[-1], zc[0], zc[-1]]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    im0 = axes[0].imshow(raster_hz, aspect='auto', origin='lower',
                         extent=ext, cmap='hot', interpolation='nearest')
    plt.colorbar(im0, ax=axes[0], label='|Nz|', shrink=0.8)
    axes[0].set_ylabel("Z (m)")
    axes[0].set_title("Horizontal Joint Score (|Nz|)")

    im1 = axes[1].imshow(raster_vt_pos, aspect='auto', origin='lower',
                         extent=ext, cmap='hot', interpolation='nearest')
    plt.colorbar(im1, ax=axes[1], label='Nx+', shrink=0.8)
    axes[1].set_ylabel("Z (m)")
    axes[1].set_title("Vertical Joint Score (Nx+, right-facing)")

    im2 = axes[2].imshow(raster_vt_neg, aspect='auto', origin='lower',
                         extent=ext, cmap='hot', interpolation='nearest')
    plt.colorbar(im2, ax=axes[2], label='Nx-', shrink=0.8)
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Z (m)")
    axes[2].set_title("Vertical Joint Score (Nx-, left-facing)")

    # Highlight which raster is active
    active_map = {'horizontal': 0, 'vertical_pos': 1, 'vertical_neg': 2}
    active_idx = active_map.get(mode, 0)
    for i, ax in enumerate(axes):
        if i == active_idx:
            for spine in ax.spines.values():
                spine.set_edgecolor('#7fbbf0')
                spine.set_linewidth(2)

    fig.tight_layout()
    return fig_to_base64(fig)


def make_profiles_plot(raster, x_edges, z_edges, detections, params, mode):
    if not detections:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return fig_to_base64(fig)

    zc = (z_edges[:-1] + z_edges[1:]) / 2
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    res = params['raster_resolution']
    sigma = params['gaussian_sigma']

    n_samples = min(6, len(detections))
    indices = np.linspace(0, len(detections)-1, n_samples, dtype=int)

    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 6), sharey=True)
    if n_samples == 1:
        axes = [axes]

    wc = max(1, int(params['window_width'] / res))

    for ax, idx in zip(axes, indices):
        wc_val, pp, ps = detections[idx]

        if mode == 'horizontal':
            # Window along X, profile along Z
            ci = np.argmin(np.abs(xc - wc_val))
            cs = max(0, ci - wc // 2)
            ce = min(raster.shape[1], cs + wc)
            profile = np.mean(raster[:, cs:ce], axis=1)
            sm = gaussian_filter1d(profile, sigma=sigma)
            ax.plot(profile, zc, 'gray', alpha=0.4, linewidth=0.8, label='Raw')
            ax.plot(sm, zc, 'steelblue', linewidth=1.5, label='Smoothed')
            ax.scatter(ps, pp, color='red', s=50, zorder=5, label='Peaks')
            ax.set_title(f"X={wc_val:.1f}m", fontsize=9)
            ax.set_xlabel("Score", fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel("Z (m)")
        else:
            # Window along Z, profile along X
            ci = np.argmin(np.abs(zc - wc_val))
            cs = max(0, ci - wc // 2)
            ce = min(raster.shape[0], cs + wc)
            profile = np.mean(raster[cs:ce, :], axis=0)
            sm = gaussian_filter1d(profile, sigma=sigma)
            ax.plot(xc, profile, 'gray', alpha=0.4, linewidth=0.8, label='Raw')
            ax.plot(xc, sm, 'steelblue', linewidth=1.5, label='Smoothed')
            ax.scatter(pp, ps, color='red', s=50, zorder=5, label='Peaks')
            ax.set_title(f"Z={wc_val:.1f}m", fontsize=9)
            ax.set_ylabel("Score", fontsize=8)
            if ax is axes[0]:
                ax.set_xlabel("X (m)")

        if ax is axes[0]:
            ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    direction = "Horizontal" if mode == 'horizontal' else "Vertical"
    fig.suptitle("%s Joint Profiles" % direction, fontsize=11)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_tracked_plot(raster, x_edges, z_edges, tracks, splines, params, mode):
    xc = (x_edges[:-1] + x_edges[1:]) / 2
    zc = (z_edges[:-1] + z_edges[1:]) / 2
    ext = [xc[0], xc[-1], zc[0], zc[-1]]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(raster, aspect='auto', origin='lower', extent=ext,
              cmap='hot', interpolation='nearest', alpha=0.6)

    jt = [t for t in tracks if t.get('label') == 'joint']
    ct = [t for t in tracks if t.get('label') == 'crack']

    jcolors = plt.cm.winter(np.linspace(0, 1, max(len(jt), 1)))
    si = 0
    for i, t in enumerate(jt):
        c = jcolors[i % len(jcolors)]
        ax.scatter(t['x'], t['z'], color=c, s=15, zorder=4,
                   edgecolors='white', linewidths=0.3)
        if si < len(splines) and splines[si]['ok']:
            ax.plot(splines[si]['x'], splines[si]['z'], color=c, lw=2, zorder=5)
        si += 1

    for t in ct:
        ax.scatter(t['x'], t['z'], color='red', s=15, zorder=4,
                   edgecolors='white', linewidths=0.3, marker='x')

    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='teal',
               markersize=8, label='Joint (%d)' % len(jt)),
        Line2D([0], [0], marker='x', color='red', lw=0,
               markersize=8, label='Crack (%d)' % len(ct)),
    ]
    ax.legend(handles=legend, fontsize=10, loc='upper right')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    direction = "Horizontal" if mode == 'horizontal' else "Vertical"
    ax.set_title("Tracked %s Joints (%d joints, %d cracks)" % (
        direction, len(jt), len(ct)))
    fig.tight_layout()
    return fig_to_base64(fig)


# ── Flask app ───────────────────────────────────────────────────────────────

app = Flask(__name__)

# Global state — populated on startup
STATE = {
    'points': None,
    'normals': None,
}


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'joint_tuner.html')


@app.route('/defaults')
def defaults():
    return jsonify(DEFAULTS)


@app.route('/info')
def info():
    pts = STATE['points']
    return jsonify({
        'n_points': len(pts),
        'x_range': [float(pts[:, 0].min()), float(pts[:, 0].max())],
        'z_range': [float(pts[:, 2].min()), float(pts[:, 2].max())],
    })


@app.route('/update', methods=['POST'])
def update():
    params = request.json
    mode = params.pop('mode', 'horizontal')

    pts = STATE['points']
    normals = STATE['normals']

    raster_hz, raster_vt_pos, raster_vt_neg, xe, ze = rasterize(pts, normals, params['raster_resolution'])

    # Pick the right raster and detection function based on mode
    if mode == 'vertical_pos':
        active_raster = raster_vt_pos
        detections = detect_peaks_vertical(active_raster, xe, ze, params)
    elif mode == 'vertical_neg':
        active_raster = raster_vt_neg
        detections = detect_peaks_vertical(active_raster, xe, ze, params)
    else:
        active_raster = raster_hz
        detections = detect_peaks_horizontal(active_raster, xe, ze, params)

    raw_tracks = track_adjacent(detections, params)
    tracks_xz = tracks_to_xz(raw_tracks, mode)
    tracks_xz = classify_tracks(tracks_xz, params)
    jt = [t for t in tracks_xz if t['label'] == 'joint']
    splines = fit_splines(jt, params)

    n_det = sum(len(d[1]) for d in detections)
    n_joints = len(jt)
    n_cracks = len([t for t in tracks_xz if t['label'] == 'crack'])

    return jsonify({
        'raster': make_raster_plot(raster_hz, raster_vt_pos, raster_vt_neg, xe, ze, params, mode),
        'profiles': make_profiles_plot(active_raster, xe, ze, detections, params, mode),
        'tracked': make_tracked_plot(active_raster, xe, ze, tracks_xz, splines, params, mode),
        'stats': {
            'n_detections': n_det,
            'n_joints': n_joints,
            'n_cracks': n_cracks,
            'n_tracks_total': len(tracks_xz),
            'raster_shape': list(active_raster.shape),
            'mode': mode,
        }
    })


def main():
    files = glob.glob("outputs/point_clouds/unrolled/displacement_*.ply")
    if not files:
        print("No displacement PLY files found in outputs/point_clouds/unrolled/")
        sys.exit(1)

    file = files[0]
    wall_id = os.path.basename(file).split("_")[1]
    print(f"Loading wall {wall_id}: {file}")

    pc = o3d.t.io.read_point_cloud(file)
    points = pc.point.positions.numpy()
    print(f"  Full cloud: {len(points)} points, "
          f"X=[{points[:, 0].min():.1f}, {points[:, 0].max():.1f}]")

    # Always crop to debug region
    center_x = DEBUG_CENTER
    if center_x is None:
        center_x = (points[:, 0].min() + points[:, 0].max()) / 2
    x_range = DEBUG_RANGE

    mask = ((points[:, 0] >= center_x - x_range) &
            (points[:, 0] <= center_x + x_range))
    points = points[mask]
    print(f"  Debug region: X = {center_x:.1f} +/- {x_range:.1f}m -> {len(points)} points")

    if len(points) == 0:
        print("No points in debug region!")
        sys.exit(1)

    print(f"  Computing normals (knn={NORMAL_KNN})...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    normals = np.asarray(pcd.normals)
    print("  Normals computed. Starting server...")

    STATE['points'] = points
    STATE['normals'] = normals

    print("\n  Open http://localhost:5555 in your browser\n")
    app.run(host='0.0.0.0', port=5555, debug=False)


if __name__ == '__main__':
    main()
