"""
Configuration file for Hand Sign Recognition System
Contains all settings for camera, MediaPipe detection, and recognition
"""

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_INDEX = 0          # Default laptop webcam (0 = first camera)
CAMERA_WARMUP_TIME = 2    # Seconds to wait for camera to initialize
FRAME_WIDTH = 640         # Camera frame width
FRAME_HEIGHT = 480        # Camera frame height

# ============================================================================
# MEDIAPIPE HAND DETECTION SETTINGS
# ============================================================================
MIN_DETECTION_CONFIDENCE = 0.7   # Minimum confidence for hand detection (0.0-1.0)
MIN_TRACKING_CONFIDENCE = 0.6    # Minimum confidence for hand tracking (0.0-1.0)
MAX_NUM_HANDS = 1                # Maximum number of hands to detect (1 for simplicity)
MODEL_COMPLEXITY = 1             # 0 = Lite, 1 = Full (more accurate)

# ============================================================================
# GESTURE RECOGNITION SETTINGS
# ============================================================================
GESTURE_CONFIDENCE_THRESHOLD = 0.75    # Minimum confidence for gesture classification
STABILITY_FRAMES = 3                   # Number of consecutive frames needed to confirm gesture
COOLDOWN_FRAMES = 5                    # Frames to wait before recognizing same gesture again

# ============================================================================
# DATA PATHS
# ============================================================================
GESTURE_MAPPINGS_PATH = "data/gesture_mappings.json"  # Path to character mappings

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================
SHOW_CAMERA_PREVIEW = True     # Show live camera feed during recognition
SHOW_LANDMARKS = True           # Draw hand landmarks on preview
SHOW_FPS = True                 # Show frames per second counter
WINDOW_NAME = "Hand Sign Recognition"

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
CONSOLE_OUTPUT = True           # Print recognized characters to console
OUTPUT_DELAY_MS = 500           # Minimum delay between outputting same character (ms)

# ============================================================================
# TEXT-TO-SPEECH SETTINGS
# ============================================================================
ENABLE_TTS = True           # Enable text-to-speech output for recognized signs
TTS_RATE = 150              # Speech rate (words per minute, default: 150)
TTS_VOLUME = 1.0            # Volume level (0.0 to 1.0)

# ============================================================================
# DEBUG SETTINGS
# ============================================================================
DEBUG_MODE = False              # Enable detailed debug logging
SAVE_DEBUG_FRAMES = False       # Save frames when gesture is recognized
