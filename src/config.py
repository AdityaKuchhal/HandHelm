import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
TESTS_DIR = os.path.join(BASE_DIR, 'tests')

# Hand detection and model parameters
# These parameters can be adjusted based on your project needs.
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Video capture settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 30

# Gesture recognition settings
GESTURE_CONFIDENCE_THRESHOLD = 0.8
GESTURE_HOLD_TIME = 0.5  # seconds to hold gesture before action
MAX_GESTURE_HISTORY = 10  # frames to keep for gesture smoothing

# Training parameters for gesture classification model
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# Data collection settings
SAMPLES_PER_GESTURE = 100
GESTURE_COLLECTION_DELAY = 0.5  # seconds between samples

# Model saving parameters
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'gesture_classifier.h5')
SCALER_SAVE_PATH = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
LABEL_ENCODER_SAVE_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

# Supported gestures (can be expanded)
SUPPORTED_GESTURES = [
    'fist',
    'open_palm', 
    'thumbs_up',
    'thumbs_down',
    'peace_sign',
    'ok_sign',
    'point_up',
    'point_down',
    'point_left',
    'point_right'
]

# Gesture to action mapping (default configuration)
DEFAULT_GESTURE_ACTIONS = {
    'fist': 'pause_media',
    'open_palm': 'play_media',
    'thumbs_up': 'volume_up',
    'thumbs_down': 'volume_down',
    'peace_sign': 'next_slide',
    'ok_sign': 'previous_slide',
    'point_up': 'scroll_up',
    'point_down': 'scroll_down',
    'point_left': 'previous_tab',
    'point_right': 'next_tab'
}

if __name__ == '__main__':
    print("Configuration Settings:")
    print("=" * 50)
    print("Directories:")
    print(f"  BASE_DIR: {BASE_DIR}")
    print(f"  RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"  PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"  MODELS_DIR: {MODELS_DIR}")
    print(f"  SCRIPTS_DIR: {SCRIPTS_DIR}")
    print(f"  TESTS_DIR: {TESTS_DIR}")
    print("\nModel Paths:")
    print(f"  MODEL_SAVE_PATH: {MODEL_SAVE_PATH}")
    print(f"  SCALER_SAVE_PATH: {SCALER_SAVE_PATH}")
    print(f"  LABEL_ENCODER_SAVE_PATH: {LABEL_ENCODER_SAVE_PATH}")
    print("\nSupported Gestures:")
    for i, gesture in enumerate(SUPPORTED_GESTURES, 1):
        print(f"  {i}. {gesture}")
    print(f"\nTotal gestures: {len(SUPPORTED_GESTURES)}")
