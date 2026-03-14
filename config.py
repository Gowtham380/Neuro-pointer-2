"""Global configuration for Neuro Pointer."""

from pathlib import Path

PROJECT_NAME = "Neuro Pointer"
WINDOW_NAME = "Neuro Pointer | H: Hand  E: Eye  T: Training  P: Test  SPACE: Sample  Q: Quit"

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_FILE = BASE_DIR / "calibration" / "calibration_data.json"

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
TARGET_FPS = 30
MIRROR_FEED = True

# Hand tracking
HAND_MAX_HANDS = 1
HAND_MIN_DETECTION_CONFIDENCE = 0.65
HAND_MIN_TRACKING_CONFIDENCE = 0.60
ACTIVE_ZONE_OFFSET_X = 0
ACTIVE_ZONE_OFFSET_Y = 0
ACTIVE_ZONE_MARGIN = 0.20


# Eye tracking
FACE_MAX_FACES = 1
FACE_MIN_DETECTION_CONFIDENCE = 0.60
FACE_MIN_TRACKING_CONFIDENCE = 0.55

# Nose-primary eye cursor control
EYE_NOSE_LANDMARK_ID = 1
EYE_NOSE_DEADZONE = 0.03
EYE_NOSE_SMOOTHING_ALPHA = 0.30
EYE_NOSE_WEIGHT = 0.85
EYE_GAZE_MICRO_WEIGHT = 0.15
ENABLE_EYE_GAZE_MICRO_ADJUST = True

# Cursor and action behavior
CURSOR_SMOOTHING_ALPHA = 0.28
CURSOR_MIN_MOVE_PIXELS = 1.0
ACTION_COOLDOWN_SECONDS = 0.28
BLINK_COOLDOWN_SECONDS = 0.35
BLINK_CONSEC_FRAMES = 2
DRAG_HOLD_FRAMES = 6
SCROLL_CONFIRM_FRAMES = 2
SCROLL_DEADZONE = 0.01
SCROLL_MULTIPLIER = 60
VOLUME_UPDATE_INTERVAL_SECONDS = 0.08
CLICK_CONFIRM_FRAMES = 3
DOUBLE_CLICK_WINDOW_SECONDS = 0.42
HAND_SCALE_EMA_ALPHA = 0.18
HAND_JITTER_RATIO_THRESHOLD = 0.12
HAND_SCROLL_MOTION_RATIO = 0.055
HAND_DIAGNOSTIC_PRINT_SECONDS = 0.45
HAND_MIN_BRIGHTNESS = 45
HAND_MAX_BRIGHTNESS = 215
HAND_SCROLL_RESIDUAL_DAMPING = 0.70
EYE_WINK_CONFIRM_FRAMES = 3
EYE_SCROLL_CONFIRM_FRAMES = 3
EYE_DRAG_TOGGLE_FRAMES = 3
EYE_MODE_TOGGLE_COOLDOWN_SECONDS = 0.55
EYE_SQUINT_RATIO = 1.16
EYE_OPEN_RATIO = 1.14
EYE_SCROLL_MOTION_RATIO = 0.018
EYE_SCROLL_GAIN = 2.2
TEST_MODE_PRINT_SECONDS = 2.0

# Feature toggles
ENABLE_VOLUME_GESTURE = True
SHOW_LANDMARKS = True
SHOW_DEBUG = True

# Default calibration values
DEFAULT_CALIBRATION = {
    "hand_region": [0.08, 0.92, 0.12, 0.90],  # min_x, max_x, min_y, max_y
    "pinch_click_threshold": 0.055,
    "right_click_threshold": 0.050,
    "fist_threshold": 0.205,
    "scroll_sensitivity": 1.00,
    "blink_ear_threshold": 0.195,
    # `gaze_center` is intentionally reused as eye neutral center for nose mode,
    # so existing key handling (C-key calibration) keeps working without main.py changes.
    "gaze_center": [0.50, 0.50],
    "gaze_sensitivity": [2.40, 2.20],
    "hand_smoothing_alpha": 0.28,
    "eye_smoothing_alpha": 1.00,
    "eye_scroll_sensitivity": 1.00,
    "nose_deadzone": 0.03,
    "nose_smoothing_alpha": 0.30,
    "nose_weight": 0.85,
    "gaze_micro_weight": 0.15,
    "volume_pinch_threshold": 0.105,
}

TRAINING_SAMPLES_PER_STAGE = 10
