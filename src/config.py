import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Hand detection and model parameters
# These parameters can be adjusted based on your project needs.
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Training parameters for gesture classification model
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Model saving parameters
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'gesture_classifier.h5')

if __name__ == '__main__':
    print("Configuration Settings:")
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)
    print("MODELS_DIR:", MODELS_DIR)
    print("Model Save Path:", MODEL_SAVE_PATH)
