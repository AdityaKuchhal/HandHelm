#!/usr/bin/env python3
"""
Generate sample gesture data for testing the preprocessing pipeline.

This script creates synthetic hand landmark data to test the data preprocessing
and model training functionality without requiring actual webcam data collection.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RAW_DATA_DIR, SUPPORTED_GESTURES

def generate_synthetic_landmarks(gesture_name, num_samples=50):
    """
    Generate synthetic hand landmarks for a given gesture.
    
    Args:
        gesture_name (str): Name of the gesture
        num_samples (int): Number of samples to generate
    
    Returns:
        list: List of sample data dictionaries
    """
    samples = []
    
    # Base landmark positions for different gestures
    gesture_templates = {
        'fist': {
            'fingers_closed': [4, 8, 12, 16, 20],  # All fingertips
            'fingers_bent': [3, 6, 10, 14, 18],    # Finger joints
        },
        'open_palm': {
            'fingers_extended': [4, 8, 12, 16, 20],
            'fingers_straight': [3, 6, 10, 14, 18],
        },
        'thumbs_up': {
            'thumb_up': [4],
            'other_fingers_closed': [8, 12, 16, 20],
        },
        'thumbs_down': {
            'thumb_down': [4],
            'other_fingers_closed': [8, 12, 16, 20],
        },
        'peace_sign': {
            'index_middle_up': [8, 12],
            'other_fingers_closed': [4, 16, 20],
        },
        'ok_sign': {
            'thumb_index_touch': [4, 8],
            'other_fingers_extended': [12, 16, 20],
        },
        'point_up': {
            'index_up': [8],
            'other_fingers_closed': [4, 12, 16, 20],
        },
        'point_down': {
            'index_down': [8],
            'other_fingers_closed': [4, 12, 16, 20],
        },
        'point_left': {
            'index_left': [8],
            'other_fingers_closed': [4, 12, 16, 20],
        },
        'point_right': {
            'index_right': [8],
            'other_fingers_closed': [4, 12, 16, 20],
        }
    }
    
    template = gesture_templates.get(gesture_name, gesture_templates['open_palm'])
    
    for i in range(num_samples):
        # Generate base landmarks (21 landmarks for a hand)
        landmarks = []
        
        # Wrist (landmark 0) - base position
        wrist_x = 320 + np.random.normal(0, 20)
        wrist_y = 240 + np.random.normal(0, 20)
        landmarks.append((wrist_x, wrist_y))
        
        # Generate finger landmarks
        for finger in range(5):  # 5 fingers
            finger_start = 1 + finger * 4  # Starting landmark for each finger
            
            for joint in range(4):  # 4 joints per finger
                landmark_idx = finger_start + joint
                
                # Base position relative to wrist
                if joint == 0:  # First joint (closest to palm)
                    base_x = wrist_x + (finger - 2) * 30 + np.random.normal(0, 5)
                    base_y = wrist_y + 20 + np.random.normal(0, 5)
                else:  # Subsequent joints
                    base_x = landmarks[-1][0] + np.random.normal(0, 3)
                    base_y = landmarks[-1][1] + np.random.normal(0, 3)
                
                # Apply gesture-specific modifications
                if landmark_idx in template.get('fingers_closed', []):
                    # Make fingers appear closed/bent
                    base_y += 10 + joint * 5
                elif landmark_idx in template.get('fingers_extended', []):
                    # Make fingers appear extended
                    base_y -= 10 - joint * 2
                elif landmark_idx in template.get('fingers_bent', []):
                    # Make joints appear bent
                    base_y += 5 + joint * 3
                elif landmark_idx in template.get('fingers_straight', []):
                    # Make joints appear straight
                    base_y -= 5 - joint * 1
                
                # Add some noise for realism
                final_x = base_x + np.random.normal(0, 2)
                final_y = base_y + np.random.normal(0, 2)
                
                landmarks.append((final_x, final_y))
        
        # Create sample data
        sample_data = {
            'gesture': gesture_name,
            'timestamp': datetime.now().isoformat(),
            'sample_id': f"{gesture_name}_{i:03d}"
        }
        
        # Add landmark coordinates
        for j, (x, y) in enumerate(landmarks):
            sample_data[f'landmark_{j}_x'] = x
            sample_data[f'landmark_{j}_y'] = y
        
        samples.append(sample_data)
    
    return samples

def generate_sample_dataset():
    """Generate a complete sample dataset for all gestures."""
    print("🎯 Generating sample gesture dataset...")
    print("=" * 50)
    
    all_samples = []
    
    for gesture in SUPPORTED_GESTURES:
        print(f"📝 Generating samples for '{gesture}'...")
        samples = generate_synthetic_landmarks(gesture, num_samples=30)
        all_samples.extend(samples)
        print(f"   ✅ Generated {len(samples)} samples")
    
    # Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sample_gesture_data_{timestamp}.csv"
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    df.to_csv(filepath, index=False)
    
    print(f"\n🎉 Sample dataset generated successfully!")
    print(f"📊 Total samples: {len(df)}")
    print(f"📁 Saved to: {filepath}")
    
    # Show distribution
    print("\n📈 Gesture distribution:")
    gesture_counts = df['gesture'].value_counts()
    for gesture, count in gesture_counts.items():
        print(f"   {gesture}: {count} samples")
    
    return filepath

def main():
    """Main function to generate sample data."""
    print("🚀 Hand Helm - Sample Data Generator")
    print("=" * 40)
    
    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    try:
        filepath = generate_sample_dataset()
        print(f"\n✅ Sample data ready for preprocessing!")
        print(f"💡 You can now run: python3 src/data_preprocessing.py")
        
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
