import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
from tqdm import tqdm

def zero_pc(pc):
    """Zero-center the point cloud (adjust as needed for your use case)"""
    points = pc.point.positions.numpy()
    points_centered = points - np.mean(points, axis=0)
    pc.point.positions = o3d.core.Tensor(points_centered)
    return pc

def compute_density_colors(points, radius=0.1, colormap='viridis'):
    """
    Compute density-based colors for each point in the point cloud.
    
    Parameters:
    -----------
    points : numpy array of shape (N, 3)
        The 3D coordinates of the points
    radius : float
        The radius within which to count neighboring points
    colormap : str
        Name of the matplotlib colormap to use
    
    Returns:
    --------
    colors : numpy array of shape (N, 3)
        RGB colors for each point based on density
    densities : numpy array of shape (N,)
        Number of neighbors for each point
    """
    
    # Build KDTree for efficient neighbor searching
    tree = KDTree(points)
    
    # Count neighbors within radius for each point
    densities = np.zeros(len(points))
    for i, point in enumerate(points):
        # Find all points within radius (including the point itself)
        neighbors = tree.query_ball_point(point, radius)
        densities[i] = len(neighbors) - 1  # Subtract 1 to exclude the point itself
    
    # Normalize densities to [0, 1] for color mapping
    min_density = densities.min()
    max_density = densities.max()
    
    if max_density > min_density:
        normalized_densities = (densities - min_density) / (max_density - min_density)
    else:
        normalized_densities = np.zeros_like(densities)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap(normalized_densities)[:, :3]  # Take only RGB, not alpha
    
    return colors, densities

def process_point_cloud_with_density(file, radius=0.1, colormap='viridis', save_path=None):
    """
    Process a single point cloud file and color it by density.
    
    Parameters:
    -----------
    file : str
        Path to the point cloud file
    radius : float
        Radius for density calculation
    colormap : str
        Colormap name for visualization
    save_path : str, optional
        If provided, save the colored point cloud to this path
    
    Returns:
    --------
    pc_colored : open3d point cloud
        The point cloud with density-based colors
    densities : numpy array
        Density values for each point
    """
    
    # Load and preprocess point cloud
    pc_source = o3d.t.io.read_point_cloud(file)
    pc_source = zero_pc(pc_source)
    points = pc_source.point.positions.numpy()
    
    # Filter to bottom 0.5 meters
    inds = points[:, 2] < 0.5
    points = points[inds]
    
    # Compute density-based colors
    colors, densities = compute_density_colors(points, radius=radius, colormap=colormap)
    
    # Create new Open3D point cloud with colored points
    pc_colored = o3d.geometry.PointCloud()
    pc_colored.points = o3d.utility.Vector3dVector(points)
    pc_colored.colors = o3d.utility.Vector3dVector(colors)
    
    # Save if path provided
    if save_path:
        o3d.io.write_point_cloud(save_path, pc_colored)
    
    return pc_colored, densities

def printf(s):
    """Print helper function"""
    print(s)

# Main processing script
if __name__ == "__main__":
    # Configuration
    target = "displacements"
    printf("Target: %s" % target)
    saveLoc = "renders/elevations/"
    
    # Density calculation parameters
    RADIUS = 0.1  # Adjust this value based on your point cloud scale
    COLORMAP = 'viridis'  # Options: 'viridis', 'plasma', 'hot', 'cool', 'turbo', etc.
    
    # Get all files
    files = glob.glob("pointClouds/unrolled/%s/*0.1.ply" % target)
    
    # Process each file
    for i in tqdm(range(len(files)), desc="Processing point clouds"):
        file = files[i]
        
        # Extract filename for saving
        filename = file.split('/')[-1].replace('.ply', '_density.ply')
        save_path = saveLoc + filename
        
        # Process point cloud with density coloring
        pc_colored, densities = process_point_cloud_with_density(
            file, 
            radius=RADIUS, 
            colormap=COLORMAP,
            save_path=save_path
        )
        
        # Print statistics for this point cloud
        if i == 0 or (i + 1) % 10 == 0:  # Print every 10th file to avoid clutter
            printf(f"File {i+1}: Min density: {densities.min():.0f}, "
                   f"Max density: {densities.max():.0f}, "
                   f"Mean density: {densities.mean():.1f}")
    
    printf("Processing complete!")
    
    # Optional: Create a colorbar legend for reference
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.5)
    
    # Create colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label('Relative Point Density (Low to High)', fontsize=12)
    
    # Save colorbar
    plt.savefig(saveLoc + 'density_colorbar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    printf(f"Colorbar saved to {saveLoc}density_colorbar.png")