"""
Density-based point cloud coloring.

Colors each point by the number of neighbors within a configurable radius,
using a KDTree for efficient spatial queries. Produces density-colored
point clouds and a reference colorbar image.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from tqdm import tqdm
from config import DENSITY_RADIUS, DENSITY_COLORMAP, DENSITY_Z_FILTER


def zero_pc(pc):
    """Zero-center the point cloud."""
    points = pc.point.positions.numpy()
    points_centered = points - np.mean(points, axis=0)
    pc.point.positions = o3d.core.Tensor(points_centered)
    return pc


def compute_density_colors(points, radius=None, colormap=None):
    """
    Compute density-based colors for each point in the point cloud.

    Parameters
    ----------
    points : numpy array of shape (N, 3)
    radius : float — neighbor search radius (default from config)
    colormap : str — matplotlib colormap name (default from config)

    Returns
    -------
    colors : numpy array of shape (N, 3) — RGB colors
    densities : numpy array of shape (N,) — neighbor counts
    """
    if radius is None:
        radius = DENSITY_RADIUS
    if colormap is None:
        colormap = DENSITY_COLORMAP

    tree = KDTree(points)

    densities = np.zeros(len(points))
    for i, point in enumerate(points):
        neighbors = tree.query_ball_point(point, radius)
        densities[i] = len(neighbors) - 1

    min_density = densities.min()
    max_density = densities.max()

    if max_density > min_density:
        normalized_densities = (densities - min_density) / (max_density - min_density)
    else:
        normalized_densities = np.zeros_like(densities)

    cmap = plt.get_cmap(colormap)
    colors = cmap(normalized_densities)[:, :3]

    return colors, densities


def process_point_cloud_with_density(file, radius=None, colormap=None, save_path=None):
    """Process a single point cloud file and color it by density."""
    if radius is None:
        radius = DENSITY_RADIUS
    if colormap is None:
        colormap = DENSITY_COLORMAP

    pc_source = o3d.t.io.read_point_cloud(file)
    pc_source = zero_pc(pc_source)
    points = pc_source.point.positions.numpy()

    inds = points[:, 2] < DENSITY_Z_FILTER
    points = points[inds]

    colors, densities = compute_density_colors(points, radius=radius, colormap=colormap)

    pc_colored = o3d.geometry.PointCloud()
    pc_colored.points = o3d.utility.Vector3dVector(points)
    pc_colored.colors = o3d.utility.Vector3dVector(colors)

    if save_path:
        o3d.io.write_point_cloud(save_path, pc_colored)

    return pc_colored, densities


def printf(s):
    print(s)


if __name__ == "__main__":
    target = "displacements"
    printf("Target: %s" % target)
    saveLoc = "renders/elevations/"

    files = glob.glob("pointClouds/unrolled/%s/*0.1.ply" % target)

    for i in tqdm(range(len(files)), desc="Processing point clouds"):
        file = files[i]

        filename = file.split('/')[-1].replace('.ply', '_density.ply')
        save_path = saveLoc + filename

        pc_colored, densities = process_point_cloud_with_density(
            file,
            radius=DENSITY_RADIUS,
            colormap=DENSITY_COLORMAP,
            save_path=save_path
        )

        if i == 0 or (i + 1) % 10 == 0:
            printf(f"File {i+1}: Min density: {densities.min():.0f}, "
                   f"Max density: {densities.max():.0f}, "
                   f"Mean density: {densities.mean():.1f}")

    printf("Processing complete!")

    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=DENSITY_COLORMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label('Relative Point Density (Low to High)', fontsize=12)

    plt.savefig(saveLoc + 'density_colorbar.png', dpi=150, bbox_inches='tight')
    plt.close()

    printf(f"Colorbar saved to {saveLoc}density_colorbar.png")
