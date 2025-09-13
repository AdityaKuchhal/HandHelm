#!/usr/bin/env python3
"""
Test script for Hand Helm data collection system.

This script tests the data collection and preprocessing functionality
to ensure everything works correctly before starting actual data collection.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gesture_detection import Hand_Detector
from src.data_collection import DataCollector
from src.data_preprocessing import DataPreprocessor
from src.config import SUPPORTED_GESTURES

def test_hand_detection():
    """Test hand detection functionality."""
    print("🧪 Testing Hand Detection...")
    
    try:
        detector = Hand_Detector()
        print("✅ Hand detector initialized successfully")
        
        # Test with a simple frame (we'll create a mock frame)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame, landmarks = detector.find_hands(test_frame)
        
        print(f"✅ Hand detection test passed")
        print(f"   Frame shape: {frame.shape}")
        print(f"   Landmarks detected: {len(landmarks) if landmarks else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hand detection test failed: {e}")
        return False

def test_data_collector():
    """Test data collector initialization."""
    print("\n🧪 Testing Data Collector...")
    
    try:
        collector = DataCollector()
        print("✅ Data collector initialized successfully")
        
        # Test camera initialization (this might fail if no camera available)
        try:
            collector.initialize_camera()
            print("✅ Camera initialization successful")
            collector.cleanup()
        except Exception as e:
            print(f"⚠️ Camera initialization failed (expected if no camera): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collector test failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing with mock data."""
    print("\n🧪 Testing Data Preprocessing...")
    
    try:
        preprocessor = DataPreprocessor()
        print("✅ Data preprocessor initialized successfully")
        
        # Create mock data
        mock_data = create_mock_data()
        
        # Test feature extraction
        features = preprocessor.extract_features(mock_data)
        print(f"✅ Feature extraction successful")
        print(f"   Features shape: {features.shape}")
        
        # Test preprocessing pipeline
        X_train, X_val, y_train, y_val = preprocessor.preprocess_data(mock_data)
        print(f"✅ Preprocessing pipeline successful")
        print(f"   Training data shape: {X_train.shape}")
        print(f"   Validation data shape: {X_val.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_mock_data():
    """Create mock gesture data for testing."""
    print("   Creating mock data for testing...")
    
    data = []
    
    for gesture in SUPPORTED_GESTURES[:3]:  # Test with first 3 gestures
        for i in range(10):  # 10 samples per gesture
            sample = {
                'gesture': gesture,
                'timestamp': f'2024-01-01T12:00:00',
                'sample_id': f'{gesture}_{i:03d}'
            }
            
            # Add mock landmark data (21 landmarks x 2 coordinates = 42 features)
            for j in range(21):
                # Generate realistic landmark coordinates
                x = np.random.randint(100, 500)
                y = np.random.randint(100, 400)
                sample[f'landmark_{j}_x'] = x
                sample[f'landmark_{j}_y'] = y
            
            data.append(sample)
    
    return pd.DataFrame(data)

def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration...")
    
    try:
        from src.config import (
            SUPPORTED_GESTURES, RAW_DATA_DIR, PROCESSED_DATA_DIR,
            SAMPLES_PER_GESTURE, GESTURE_COLLECTION_DELAY
        )
        
        print("✅ Configuration loaded successfully")
        print(f"   Supported gestures: {len(SUPPORTED_GESTURES)}")
        print(f"   Raw data directory: {RAW_DATA_DIR}")
        print(f"   Processed data directory: {PROCESSED_DATA_DIR}")
        print(f"   Samples per gesture: {SAMPLES_PER_GESTURE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directory_structure():
    """Test if all required directories exist."""
    print("\n🧪 Testing Directory Structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'models',
        'scripts',
        'tests',
        'src'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("🚀 Hand Helm - Data Collection System Tests")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_configuration),
        ("Hand Detection", test_hand_detection),
        ("Data Collector", test_data_collector),
        ("Data Preprocessing", test_data_preprocessing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Data collection system is ready.")
        print("\nNext steps:")
        print("1. Run: python src/data_collection.py")
        print("2. Collect gesture data")
        print("3. Run: python src/data_preprocessing.py")
    else:
        print("⚠️ Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    main()
