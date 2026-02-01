"""
Centralized configuration for object-inspector-3d.
All settings are defined here to avoid magic numbers in code.
"""

# =============================================================================
# Window Settings
# =============================================================================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Object Inspector 3D"
TARGET_FPS = 0  # 0 = unlimited FPS
BACKGROUND_COLOR = (25, 25, 28)  # Dark charcoal (Tesla-inspired)

# =============================================================================
# Performance Settings
# =============================================================================
USE_HARDWARE_ACCEL = True  # Use hardware acceleration
VSYNC = False  # Vertical sync (limits to monitor refresh rate)
LOW_POLY_MODE = True  # Use fewer polygons for shapes
SKIP_UI_FRAMES = 2  # Update UI every N frames (1 = every frame)

# =============================================================================
# Camera Settings
# =============================================================================
CAMERA_FOV = 60.0  # Field of view in degrees
CAMERA_NEAR = 0.1  # Near clipping plane
CAMERA_FAR = 1000.0  # Far clipping plane
CAMERA_DISTANCE = 5.0  # Initial distance from origin
CAMERA_MIN_DISTANCE = 1.0
CAMERA_MAX_DISTANCE = 50.0

# Camera Controls
ORBIT_SENSITIVITY = 0.5  # Mouse sensitivity for orbiting
PAN_SENSITIVITY = 0.01  # Mouse sensitivity for panning
ZOOM_SENSITIVITY = 0.5  # Scroll sensitivity for zooming

# Initial camera angles (in radians)
CAMERA_INITIAL_YAW = 0.5  # Horizontal angle
CAMERA_INITIAL_PITCH = 0.3  # Vertical angle

# =============================================================================
# Shape Settings
# =============================================================================
DEFAULT_SHAPE = "cube"
AVAILABLE_SHAPES = [
    # Basic primitives (keys 1-7)
    "cube", "sphere", "cylinder", "cone", "torus", "plane", "pyramid",
]

# Shape colors
SHAPE_COLOR = (100, 150, 255)  # Default fill color
SHAPE_COLOR_ALT = (80, 120, 200)  # Alternate for shading
WIREFRAME_COLOR = (200, 200, 200)
WIREFRAME_COLOR_HIGHLIGHT = (255, 255, 255)
POINT_COLOR = (255, 200, 100)
POINT_SIZE = 4

# Shape generation parameters (reduced for performance)
SPHERE_SEGMENTS = 16  # Was 24
SPHERE_RINGS = 12  # Was 16
CYLINDER_SEGMENTS = 16  # Was 24
CONE_SEGMENTS = 16  # Was 24
TORUS_MAJOR_SEGMENTS = 16  # Was 24
TORUS_MINOR_SEGMENTS = 8  # Was 12

# =============================================================================
# Lighting Settings
# =============================================================================
LIGHT_DIRECTION = (0.5, 0.8, 0.6)  # Normalized direction to light
AMBIENT_INTENSITY = 0.3  # Ambient light strength (0-1)
DIFFUSE_INTENSITY = 0.7  # Diffuse light strength (0-1)
SPECULAR_INTENSITY = 0.3  # Specular highlight strength (0-1)
SPECULAR_POWER = 32  # Shininess exponent

# =============================================================================
# ML Model Settings
# =============================================================================
ML_ENABLED = True
ML_ASYNC_PROCESSING = True  # Run ML in background thread

# Model paths (relative to models/pretrained/)
CLASSIFIER_MODEL = "mobilenet_v2.onnx"
POSE_MODEL = "pose_estimation.onnx"
DEPTH_MODEL = "midas_small.onnx"
FEATURE_MODEL = "feature_detector.onnx"

# Processing resolution (lower for speed)
ML_PROCESS_WIDTH = 320
ML_PROCESS_HEIGHT = 240

# Inference settings
ML_CONFIDENCE_THRESHOLD = 0.5
ML_MAX_DETECTIONS = 10

# Classification labels
SHAPE_LABELS = [
    # Basic primitives
    "cube", "sphere", "cylinder", "cone", "torus", "plane", "pyramid", "prism",
    "unknown"
]

# =============================================================================
# Webcam Settings
# =============================================================================
WEBCAM_ENABLED = True
WEBCAM_DEVICE_ID = 0
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
WEBCAM_FPS = 30

# =============================================================================
# Post-Processing Effects
# =============================================================================
# Glow effect
GLOW_ENABLED = True
GLOW_INTENSITY = 0.4
GLOW_RADIUS = 2
GLOW_THRESHOLD = 0.5

# Vignette effect
VIGNETTE_ENABLED = True
VIGNETTE_STRENGTH = 0.3
VIGNETTE_RADIUS = 0.8

# Scanlines (retro effect)
SCANLINES_ENABLED = False
SCANLINES_SPACING = 3
SCANLINES_INTENSITY = 0.1

# Effect presets
EFFECT_PRESETS = {
    "clean": {
        "glow": False,
        "vignette": False,
        "scanlines": False,
    },
    "subtle": {
        "glow": True,
        "glow_intensity": 0.3,
        "vignette": True,
        "vignette_strength": 0.2,
    },
    "vibrant": {
        "glow": True,
        "glow_intensity": 0.6,
        "vignette": True,
        "vignette_strength": 0.4,
    },
    "retro": {
        "glow": True,
        "vignette": True,
        "scanlines": True,
    },
}
DEFAULT_EFFECT_PRESET = "subtle"

# =============================================================================
# Visualization Settings
# =============================================================================
VISUALIZATION_MODES = ["wireframe", "solid", "points", "exploded"]
DEFAULT_VISUALIZATION_MODE = "solid"

# Transition settings
TRANSITION_DURATION = 0.5  # seconds
TRANSITION_EASING = "ease_out"

# Grid settings
SHOW_GRID = True
GRID_SIZE = 10
GRID_DIVISIONS = 10
GRID_COLOR = (50, 50, 55)
GRID_COLOR_AXIS = (80, 80, 85)

# Axis indicator
SHOW_AXES = True
AXIS_LENGTH = 1.0
AXIS_COLORS = {
    "x": (255, 100, 100),  # Red
    "y": (100, 255, 100),  # Green
    "z": (100, 100, 255),  # Blue
}

# =============================================================================
# UI Settings
# =============================================================================
PANEL_HEIGHT = 40
PANEL_COLOR = (25, 25, 30)
PANEL_ALPHA = 220

FONT_NAME = "Consolas"
FONT_SIZE = 14
FONT_SIZE_LARGE = 16
FONT_COLOR = (200, 200, 200)
FONT_COLOR_HIGHLIGHT = (255, 255, 255)
FONT_COLOR_DIM = (120, 120, 120)

# Help overlay
HELP_BACKGROUND_ALPHA = 200
HELP_PADDING = 20

# =============================================================================
# Color Palettes
# =============================================================================
COLOR_PALETTES = {
    "default": [
        (100, 150, 255),
        (150, 100, 255),
        (255, 100, 150),
        (255, 150, 100),
        (100, 255, 150),
    ],
    "cool": [
        (70, 130, 180),
        (100, 149, 237),
        (65, 105, 225),
        (138, 43, 226),
        (75, 0, 130),
    ],
    "warm": [
        (255, 99, 71),
        (255, 140, 0),
        (255, 215, 0),
        (255, 69, 0),
        (220, 20, 60),
    ],
    "neon": [
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 128),
    ],
}
DEFAULT_PALETTE = "default"

# =============================================================================
# Tesla-Style Exploded View Settings
# =============================================================================
# Accent colors (Tesla-inspired)
ACCENT_PRIMARY = (0, 180, 255)      # Electric blue
ACCENT_SECONDARY = (255, 60, 60)    # Red
ACCENT_TEAL = (0, 200, 180)         # Teal
ACCENT_GOLD = (255, 180, 50)        # Gold
ACCENT_PURPLE = (180, 100, 255)     # Purple

# Part colors for assemblies
ASSEMBLY_COLORS = [
    (0, 180, 255),    # Electric blue
    (255, 60, 60),    # Red
    (0, 200, 180),    # Teal
    (255, 180, 50),   # Gold
    (180, 100, 255),  # Purple
    (100, 255, 150),  # Green
    (255, 150, 200),  # Pink
    (200, 200, 200),  # Silver
]

# Explosion animation
EXPLOSION_ANIMATION_SPEED = 2.0     # Factor change per second
EXPLOSION_EASING = "smoothstep"     # Easing function type

# Label settings
LABEL_FONT_SIZE = 12
LABEL_BACKGROUND_COLOR = (30, 30, 35, 200)  # Semi-transparent dark
LABEL_TEXT_COLOR = (255, 255, 255)
LABEL_LEADER_COLOR = (100, 100, 110)
LABEL_DOT_COLOR = (150, 150, 160)
LABEL_DOT_RADIUS = 4
LABEL_PADDING = (8, 4)              # Horizontal, vertical padding
LABEL_BORDER_RADIUS = 6

# Slider settings
SLIDER_TRACK_COLOR = (50, 50, 55)
SLIDER_FILL_COLOR = (0, 180, 255)   # Same as accent primary
SLIDER_HANDLE_COLOR = (255, 255, 255)
SLIDER_HANDLE_RADIUS = 8
SLIDER_TRACK_HEIGHT = 4
SLIDER_WIDTH = 200
SLIDER_MARGIN = 20

# =============================================================================
# Debug Settings
# =============================================================================
DEBUG_MODE = False
SHOW_FPS = True
SHOW_STATS = True
