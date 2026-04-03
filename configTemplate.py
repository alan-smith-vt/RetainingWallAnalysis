"""
Centralized configuration for the Retaining Wall Analysis pipeline.

All hardcoded values from across the project are collected here so they
can be adjusted in one place.
"""

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
FEET_TO_METERS = 0.3048  # 1 foot in meters
METERS_TO_FEET = 3.28084

# ---------------------------------------------------------------------------
# Input data paths
# ---------------------------------------------------------------------------
POINT_CLOUD_FILE = "pointClouds/charlotteWall_Level7_clean.ply"
WALL_VERTICES_PATTERN = "pointClouds/Vertices.ply"

# ---------------------------------------------------------------------------
# Wall analysis (analysis/wall_analysis.py)
# ---------------------------------------------------------------------------
NUM_WORKERS = 0                 # Parallel workers for wall analysis (0 = auto, 1 = serial)
WALL_IDS = [1]                  # Which walls to process
ANALYSIS_SPACINGS = [1]         # Meters between cross-section slices
SEGMENT_LENGTH = 10             # Meters — data window for each piecewise slope fit
SEGMENT_OVERLAP = 0             # Meters — overlap between adjacent segments
                                # Step = SEGMENT_LENGTH - SEGMENT_OVERLAP
                                # e.g. length=2, overlap=1 → 1m steps, each fit uses
                                # 2m of data but the line is drawn over the 1m center
SLICE_OVERLAP = 0               # Meters — horizontal overlap between adjacent slices
                                # Step = ANALYSIS_SPACING - SLICE_OVERLAP
                                # e.g. spacing=1, overlap=0.8 → 0.2m steps, each fit
                                # uses 1m of data but output covers the 0.2m step
SLICE_HALF_WIDTH = 1.5          # Meters from wall base Y to include points
                                # Must exceed wall_height * batter + margin for
                                # battered walls (e.g. 11m wall at 8% needs > 0.88m)
TOP_OF_WALL_OFFSET = -0.25      # Meters — removes the lip at the top of the wall
MAX_DISPLACEMENT_FOR_COLORS = 0.3  # Meters — normalizes displacement color mapping (symmetric fallback)
MAX_DISPLACEMENT_POSITIVE = None   # Meters — max forward displacement for colors (None = use MAX_DISPLACEMENT_FOR_COLORS)
MAX_DISPLACEMENT_NEGATIVE = None   # Meters — max backward displacement for colors (None = use MAX_DISPLACEMENT_FOR_COLORS)
EXPECTED_WALL_SLOPE = 0.04      # Expected batter as Y/Z ratio (0 = vertical, 0.04 = 4%)
SLOPE_THRESHOLD = None          # Percent slope threshold (None = use jet colormap)
SLOPE_COLORMAP_RANGE = 3.5      # Maps 0 to -3.5% slope onto 0–1 for jet colormap
TOP_INCHES_FOR_NEW_SLOPE = 0.45 # Meters below z_max for averaging (≈18 inches)

# Discrete slope color ranges (percent value -> hex color)
DISCRETE_SLOPE_RANGES = [
    (float('-inf'), -0.01, '#df03fc'),  # bright purple
    (-0.01,          0.01, '#03fc24'),  # bright green
    ( 0.01,          0.02, '#03fcf4'),  # light blue
    ( 0.02,          0.03, '#2803fc'),  # dark blue
    ( 0.03,          0.04, '#fccf03'),  # light orange
    ( 0.04,          0.05, '#fc5e03'),  # dark orange
    ( 0.05,  float('inf'), '#fc0303'),  # bright red
]

# ---------------------------------------------------------------------------
# Station splits — divide wall renders into sub-images by station range
# Each entry is (start_station_ft, end_station_ft).
# Station format: 4+04 = 404 ft, 8+06.5 = 806.5 ft, etc.
# Set to None or [] to render the full wall as a single image.
# ---------------------------------------------------------------------------
STATION_SPLITS = [
    (0, 404),       # 0+00 to 4+04
    (404, 806.5),   # 4+04 to 8+06.5
    (806.5, 1050),  # 8+06.5 to 10+50
]

# ---------------------------------------------------------------------------
# Point cloud rendering (rendering/point_cloud.py)
# ---------------------------------------------------------------------------
RENDER_DPI = 10
RENDER_RESOLUTION = 100         # Pixels per meter equivalent
MARKER_SIZE_DEFAULT = 250
MARKER_SIZE_DISPLACEMENTS = 500
MARKER_SIZE_SLOPES = 1000
RENDER_TARGET = "slopes/thresholds/"  # Which target set to render

# ---------------------------------------------------------------------------
# Elevation rendering (rendering/elevation.py)
# ---------------------------------------------------------------------------
REFERENCE_ELEVATION_M = 232.1052      # Meters (≈761.5 feet)
REFERENCE_ELEVATION_FT = 761.5
ELEVATION_WINDOW_M = FEET_TO_METERS   # ±1 foot around reference
VERTICAL_EXAGGERATION = 20            # Scale factor for elevation profiles
ELEVATION_MARKER_SIZE = 500

# ---------------------------------------------------------------------------
# Elevation curve rendering (rendering/elevation_curves.py)
# ---------------------------------------------------------------------------
CURVE_MARKER_SIZE = 1000

# ---------------------------------------------------------------------------
# Density analysis (analysis/density.py)
# ---------------------------------------------------------------------------
DENSITY_RADIUS = 0.1            # Meters — KDTree neighbor search radius
DENSITY_COLORMAP = 'viridis'    # Matplotlib colormap name
DENSITY_Z_FILTER = 0.5          # Keep points with z < this value (meters)

# ---------------------------------------------------------------------------
# Curve fitting (analysis/curve_fitting.py)
# ---------------------------------------------------------------------------
CURVE_WINDOW_THICKNESS = 0.5    # Meters — sliding window thickness
CURVE_STEP_SIZE = 1             # Meters — step between windows
CURVE_BIN_WIDTH = 0.2           # Meters — density histogram bin width
CURVE_CROP_DELTA = 0.01         # Meters (±1 cm around peak)
CURVE_OUTLIER_THRESHOLD = 0.5   # Meters — residual threshold for outlier removal
CURVE_SMOOTHING_FACTOR_1 = 5    # First-pass spline smoothing
CURVE_SMOOTHING_FACTOR_2 = 1    # Second-pass spline smoothing
CURVE_NUM_SMOOTH_POINTS = 1000  # Points along the smoothed spline

# ---------------------------------------------------------------------------
# Grid and label overlay (postprocessing/grid_labels.py)
# ---------------------------------------------------------------------------
GRID_PAD_X = 305 * 2            # Pixels — left padding (2 stations)
GRID_PADDING = [0, 900, 305 * 2, 10]  # [top, bottom, left, right] in pixels
GRID_LINE_THICKNESS = 3
GRID_DELTA_X = 305              # Pixels per 10 feet horizontal
GRID_DELTA_Y = (1219 / 24) * 3 # Pixels per 3-inch vertical spacing
GRID_PIXELS_PER_FOOT = 305 / 10  # 30.5 pixels per foot
GRID_FONT_SIZE = 100
GRID_RED_LINE_ELEVATION = 761.6 # Feet — reference elevation for red line
GRID_RED_LINE_THICKNESS = 6

# ---------------------------------------------------------------------------
# Slope labels (postprocessing/slope_labels.py)
# ---------------------------------------------------------------------------
SLOPE_LABEL_FONT_SIZE = 64
SLOPE_LABEL_COLOR = (100, 100, 100)  # Dark gray RGB

# ---------------------------------------------------------------------------
# Image overlay (postprocessing/overlay.py)
# ---------------------------------------------------------------------------
OVERLAY_TRANSPARENCY = 0.5

# Per-wall overlay offsets: {wall_id: (x_offset, y_delta)}
# y_offset is computed as (base_height - overlay_height) - y_delta
WALL_OVERLAY_OFFSETS = {
    1: (180, 416),
    2: (180, 319),
    3: (150, 308),
}

# ---------------------------------------------------------------------------
# Downsampling (utils/downsample.py)
# ---------------------------------------------------------------------------
DOWNSAMPLE_VOXEL_SIZE = 0.04    # Meters (4 cm)
DOWNSAMPLE_INPUT = "KCLN Point Cloud - GZA.ply"
DOWNSAMPLE_OUTPUT = "PC_down_04.ply"

# ---------------------------------------------------------------------------
# Scale template (utils/scale_template.py)
# ---------------------------------------------------------------------------
SCALE_TEMPLATE_SIZE = 305       # Pixels (width and height)

# ---------------------------------------------------------------------------
# Wall drawing scale factors (utils/view.py)
# ---------------------------------------------------------------------------
WALL_SCALE_FACTORS = {
    1: 305 / 328,   # ≈ 0.9299
    2: 305 / 418,   # ≈ 0.7297
    3: 305 / 446,   # ≈ 0.6835
}

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_DIR_DISPLACEMENTS = "pointClouds/displacements"
OUTPUT_DIR_SLOPES = "pointClouds/slopes"
OUTPUT_DIR_NEW_SLOPES = "pointClouds/new_slopes"
OUTPUT_DIR_UNROLLED = "pointClouds/unrolled"
OUTPUT_DIR_CROSS_SECTIONS = "pointClouds/unrolled/crossSections"
OUTPUT_DIR_RENDERS = "renders"
OUTPUT_DIR_ELEVATIONS = "renders/elevations"
OUTPUT_DIR_ELEVATION_CURVES = "renders/elevations/curves"
OUTPUT_DIR_SLOPE_RENDERS = "renders/slopes"
OUTPUT_DIR_SLOPE_LABELED = "renders/slopes/labeled"
OUTPUT_DIR_OVERLAYS = "renders/overlays/elevations"
OUTPUT_DIR_EXCEL = "excel_data"
