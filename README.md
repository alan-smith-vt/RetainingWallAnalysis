# Retaining Wall Analysis

A point-cloud-based analysis pipeline for evaluating structural displacement, slope, and settlement characteristics of concrete retaining walls. The system processes 3D LiDAR point cloud data, computes geometric metrics, and generates publication-ready engineering visualizations.

## Overview

This tool takes a 3D point cloud of a retaining wall site along with wall centerline vertices, then:

1. **Extracts cross-sections** perpendicular to each wall segment at configurable intervals.
2. **Fits piecewise linear slopes** to each cross-section to measure wall lean/displacement.
3. **Colors points** by slope magnitude (jet colormap) and vertical displacement from a reference baseline.
4. **Unrolls** the 3D wall into a 2D representation for easier visualization.
5. **Generates elevation profiles** at a reference elevation with vertical exaggeration.
6. **Fits settlement curves** using density-peak extraction and spline smoothing.
7. **Overlays results** onto engineering drawings with grid lines, labels, and legends.

## Project Structure

```
RetainingWallAnalysis/
├── config.py                          # All configurable settings in one place
├── requirements.txt                   # Python dependencies
├── README.md
│
├── analysis/                          # Core analysis algorithms
│   ├── wall_analysis.py               # Main wall geometry analysis engine
│   ├── density.py                     # Density-based point cloud coloring
│   └── curve_fitting.py               # Settlement curve extraction via spline fitting
│
├── rendering/                         # Visualization / image generation
│   ├── point_cloud.py                 # 3D point cloud → 2D scatter plot images
│   ├── elevation.py                   # Elevation profile renderer
│   ├── elevation_curves.py            # Settlement curve renderer
│   └── colorbar.py                    # Slope and displacement legend generator
│
├── postprocessing/                    # Image annotation and compositing
│   ├── grid_labels.py                 # Grid lines and elevation labels
│   ├── slope_labels.py                # Slope percentage text labels
│   ├── overlay.py                     # Composite analysis onto engineering drawings
│   └── stack_images.py                # Vertically stack wall images
│
├── utils/                             # Helper scripts
│   ├── downsample.py                  # Voxel downsampling for large point clouds
│   ├── scale_template.py              # Scale reference image generator
│   └── view.py                        # Wall drawing rescaler and image viewer
│
├── pointClouds/                       # Input/output point cloud data (not tracked)
│   ├── KCLN_noColorCrop2_4cm.ply      # Main downsampled point cloud
│   ├── wall_[1-3]_vertices_all.ply    # Wall centerline vertices
│   ├── displacements/                 # Displacement-colored point clouds
│   ├── slopes/                        # Slope-colored point clouds
│   ├── new_slopes/                    # Alternative slope algorithm outputs
│   └── unrolled/                      # 2D-unrolled point clouds
│       ├── displacements/
│       ├── slopes/
│       ├── new_slopes/
│       ├── crossSections/
│       └── elevations/curves/
│
├── renders/                           # Output images (not tracked)
│   ├── elevations/                    # Elevation profile images
│   │   └── curves/                    # Settlement curve images
│   ├── slopes/                        # Slope visualization images
│   │   └── labeled/                   # Slope images with % labels
│   └── overlays/elevations/           # Final composited images
│
├── excel_data/                        # CSV exports (not tracked)
│   └── wall_[1-3]_settlement.csv      # Settlement profile data
│
└── pdfs/                              # Source engineering drawings (not tracked)
    └── wall_[1-3].png
```

## Pipeline Workflow

Run each step in order. Each script is standalone and can be executed directly.

### Step 1 — Downsample (optional)

```bash
python utils/downsample.py
```

Reduces the raw point cloud to a 4 cm voxel grid for faster processing.

### Step 2 — Wall Analysis

```bash
python analysis/wall_analysis.py
```

The core engine. For each wall and spacing:
- Rotates wall segments to align with the x-axis.
- Extracts perpendicular cross-sections within 0.5 m of the centerline.
- Fits piecewise linear slopes in 10 m segments.
- Colors points by slope (jet colormap, 0 to -3.5%) and displacement.
- Outputs colored `.ply` files and slope CSVs.

### Step 3 — Render Point Clouds to Images

```bash
python rendering/point_cloud.py
```

Converts slope/displacement point clouds into 2D scatter-plot PNG images.

### Step 4 — Elevation Profiles

```bash
python rendering/elevation.py
```

Extracts a horizontal slice at the reference elevation (761.5 ft / 232.1 m) with +/-1 foot window and applies 20x vertical exaggeration.

### Step 5 — Settlement Curves

```bash
python analysis/curve_fitting.py
python rendering/elevation_curves.py
```

Finds peak-density points along the elevation slice, fits a smoothing spline, removes outliers, and exports settlement values in feet to CSV.

### Step 6 — Density Coloring (optional)

```bash
python analysis/density.py
```

Colors points by local neighbor density using a KDTree search.

### Step 7 — Add Grid Lines and Labels

```bash
python postprocessing/grid_labels.py
python postprocessing/slope_labels.py
```

Overlays station grid lines, elevation labels, a red reference line at 761.6 ft, and slope percentage annotations.

### Step 8 — Composite onto Engineering Drawings

```bash
python postprocessing/overlay.py
```

Blends analysis results onto the scaled engineering wall drawings at 50% transparency.

### Step 9 — Final Assembly

```bash
python postprocessing/stack_images.py
```

Vertically stacks all wall images into a single combined output.

### Step 10 — Legends

```bash
python rendering/colorbar.py
```

Generates color legend images for slope percentages and displacement scales.

## Configuration

All analysis parameters are centralized in **`config.py`**. Key settings include:

| Parameter | Default | Description |
|---|---|---|
| `WALL_IDS` | `[1]` | Which walls to process |
| `ANALYSIS_SPACINGS` | `[1]` | Meters between cross-section slices |
| `SEGMENT_LENGTH` | `10` | Meters per piecewise slope segment |
| `SLICE_HALF_WIDTH` | `0.5` | Meters from centerline to include |
| `REFERENCE_ELEVATION_M` | `232.1052` | Reference elevation in meters (761.5 ft) |
| `ELEVATION_WINDOW_M` | `0.3048` | +/- window around reference (1 foot) |
| `VERTICAL_EXAGGERATION` | `20` | Scale factor for elevation profiles |
| `SLOPE_COLORMAP_RANGE` | `3.5` | Maps 0 to -3.5% slope onto jet colormap |
| `MAX_DISPLACEMENT_FOR_COLORS` | `0.3` | Max displacement (m) for color normalization |
| `DENSITY_RADIUS` | `0.1` | KDTree neighbor search radius (m) |
| `OVERLAY_TRANSPARENCY` | `0.5` | Blend opacity for drawing overlays |
| `DOWNSAMPLE_VOXEL_SIZE` | `0.04` | Voxel size for downsampling (m) |

See `config.py` for the complete list of parameters including rendering, grid, label, and per-wall overlay settings.

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- **open3d** — Point cloud I/O and processing
- **numpy** — Numerical computing
- **matplotlib** — Plotting and colormaps
- **opencv-python** — Image processing
- **scipy** — Spatial queries (KDTree) and spline fitting
- **Pillow** — Text rendering on images
- **tqdm** — Progress bars

## Input Data

Place your data in the expected directories before running:

- `pointClouds/KCLN_noColorCrop2_4cm.ply` — Main downsampled point cloud
- `pointClouds/wall_1_vertices_all.ply` (and wall_2, wall_3) — Wall centerline vertices
- `pdfs/wall_1.png` (and wall_2, wall_3) — Engineering wall drawings

## Output Products

| Type | Location | Format |
|---|---|---|
| Displacement point clouds | `pointClouds/displacements/` | `.ply` |
| Slope point clouds | `pointClouds/slopes/` | `.ply` |
| Unrolled point clouds | `pointClouds/unrolled/` | `.ply` |
| Settlement curves | `pointClouds/unrolled/elevations/curves/` | `.ply` |
| Elevation profile images | `renders/elevations/` | `.png` |
| Slope images with labels | `renders/slopes/labeled/` | `.png` |
| Engineering drawing composites | `renders/overlays/elevations/` | `.jpg` |
| Combined wall image | `renders/overlays/elevations/combined.png` | `.png` |
| Settlement data | `excel_data/wall_*_settlement.csv` | `.csv` |
| Slope data | `renders/slopes/slope_*.csv` | `.csv` |
