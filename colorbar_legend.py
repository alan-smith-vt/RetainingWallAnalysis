import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def hex_to_rgb(hex_color):
    """Convert hex color to RGB array."""
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)])

def value_to_rgb(percent):
    if percent > 0:
        return np.array([0.0, 0.0, 0.0, 0.5])  # Black with 50% transparency
    cmap = plt.cm.jet
    percent = (-percent*100)/3.5  # map 0 to -3.5% as 0 to 1 for cmap
    percent = min(1.0, percent)  # Clamp to maximum of 1.0
    rgba = cmap(percent)  # Get RGBA from colormap
    return np.array([rgba[0], rgba[1], rgba[2], 0.5])  # Set alpha to 0.5

def displacement_to_rgb(inches):
    """Map displacement values (0 to 12 inches) to jet colormap with 50% transparency."""
    if inches < 0:
        inches = 0
    elif inches > 12:
        inches = 12
    
    cmap = plt.cm.jet
    normalized = inches / 12.0  # map 0 to 12 inches as 0 to 1 for cmap
    rgba = cmap(normalized)  # Get RGBA from colormap
    return np.array([rgba[0], rgba[1], rgba[2], 0.5])  # Set alpha to 0.5

def create_slope_colorbar():
    """Create a discrete legend for slope percentages."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate ranges and colors based on the actual value_to_rgb function
    ranges_labels = []
    
    # Positive values (all get black color)
    positive_color = value_to_rgb(0.01)  # Any positive value
    ranges_labels.append(("> 0%", positive_color))
    
    # Negative values using the jet colormap - updated to match -3.5% range
    negative_ranges = [
        ("0% to -0.5%", -0.0025),    # Middle of 0 to -0.5%
        ("-0.5% to -1.0%", -0.0075), # Middle of -0.5% to -1.0%
        ("-1.0% to -1.5%", -0.0125), # Middle of -1.0% to -1.5%
        ("-1.5% to -2.0%", -0.0175), # Middle of -1.5% to -2.0%
        ("-2.0% to -2.5%", -0.0225), # Middle of -2.0% to -2.5%
        ("-2.5% to -3.0%", -0.0275), # Middle of -2.5% to -3.0%
        ("-3.0% to -3.5%", -0.0325), # Middle of -3.0% to -3.5%
        ("< -3.5%", -0.035)          # At -3.5% limit
    ]
    
    # Generate colors for negative ranges
    for label, sample_value in negative_ranges:
        color = value_to_rgb(sample_value)
        ranges_labels.append((label, color))
    
    # Create legend patches with transparency
    patches = [mpatches.Patch(color=color[:3], alpha=color[3], label=label) for label, color in ranges_labels]
    
    # Create the legend
    legend = ax.legend(handles=patches, loc='center', title='Slope Percentages', 
                      title_fontsize=12, fontsize=10)
    ax.axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.show()

def create_displacement_colorbar():
    """Create a discrete legend for displacement values (0 to 12 inches)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate ranges and colors for displacement
    ranges_labels = []
    
    displacement_ranges = [
        ("≤ 0\"", 0),     # Middle of 0 to 2 inches
        ("2\"", 2.0),     # Middle of 2 to 4 inches
        ("4\"", 4.0),     # Middle of 4 to 6 inches
        ("6\"", 6.0),     # Middle of 6 to 8 inches
        ("8\"", 8.0),    # Middle of 8 to 10 inches
        ("10\"", 10.0),  # Middle of 10 to 12 inches
        ("≥ 12\"", 12.0)         # At 12 inch limit
    ]
    
    # Generate colors for displacement ranges
    for label, sample_value in displacement_ranges:
        color = displacement_to_rgb(sample_value)
        ranges_labels.append((label, color))
    
    # Create legend patches with transparency
    patches = [mpatches.Patch(color=color[:3], alpha=color[3], label=label) for label, color in ranges_labels]
    
    # Create the legend
    legend = ax.legend(handles=patches, loc='center', title='Displacement (in)', 
                      title_fontsize=12, fontsize=10)
    ax.axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.show()

# Run the functions
if __name__ == "__main__":
    print("Slope Percentages Legend:")
    create_slope_colorbar()
    
    print("Displacement Legend:")
    create_displacement_colorbar()