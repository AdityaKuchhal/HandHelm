import cv2
import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SUPPORTED_GESTURES,
    SAMPLES_PER_GESTURE, GESTURE_COLLECTION_DELAY,
    CAMERA_WIDTH, CAMERA_HEIGHT, FPS
)
from src.gesture_detection import Hand_Detector

class DataCollector:
    """
    A class to collect hand gesture training data using webcam.
    
    This class provides functionality to:
    - Capture hand gestures in real-time
    - Extract hand landmarks using MediaPipe
    - Store data in organized CSV format
    - Provide user-friendly collection interface
    """
    
    def __init__(self):
        self.detector = Hand_Detector()
        self.cap = None
        self.collected_data = []
        self.current_gesture = None
        self.sample_count = 0
        self.collection_start_time = None
        
        # Create directories if they don't exist
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
    def initialize_camera(self):
        """Initialize the webcam with specified settings."""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam.")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        print("✅ Camera initialized successfully!")
        return True
    
    def collect_gesture_data(self, gesture_name, num_samples=None):
        """
        Collect training data for a specific gesture.
        
        Args:
            gesture_name (str): Name of the gesture to collect
            num_samples (int): Number of samples to collect (default from config)
        """
        if gesture_name not in SUPPORTED_GESTURES:
            print(f"❌ Error: '{gesture_name}' is not a supported gesture.")
            print(f"Supported gestures: {', '.join(SUPPORTED_GESTURES)}")
            return False
        
        if num_samples is None:
            num_samples = SAMPLES_PER_GESTURE
            
        self.current_gesture = gesture_name
        self.sample_count = 0
        self.collection_start_time = time.time()
        
        print(f"\n🎯 Collecting data for '{gesture_name}'")
        print(f"📊 Target samples: {num_samples}")
        print("📝 Instructions:")
        print("   - Show the gesture clearly in the camera")
        print("   - Move your hand slightly for variety")
        print("   - Press 'c' to capture a sample")
        print("   - Press 'q' to quit collection")
        print("   - Press 's' to skip this gesture")
        
        while self.sample_count < num_samples:
            success, frame = self.cap.read()
            if not success:
                print("❌ Error: Failed to capture frame.")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands and draw landmarks
            frame, landmark_data = self.detector.find_hands(frame)
            
            # Add collection info overlay
            self._draw_collection_info(frame, gesture_name, num_samples)
            
            # Show frame
            cv2.imshow("Hand Helm - Data Collection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Collection stopped by user.")
                break
            elif key == ord('s'):
                print(f"⏭️ Skipping '{gesture_name}' collection.")
                break
            elif key == ord('c'):
                if landmark_data:
                    self._save_sample(landmark_data[0], gesture_name)
                    self.sample_count += 1
                    print(f"✅ Sample {self.sample_count}/{num_samples} captured!")
                    
                    # Add delay between samples
                    time.sleep(GESTURE_COLLECTION_DELAY)
                else:
                    print("⚠️ No hand detected! Try again.")
        
        if self.sample_count > 0:
            print(f"🎉 Collected {self.sample_count} samples for '{gesture_name}'")
        
        return self.sample_count > 0
    
    def _draw_collection_info(self, frame, gesture_name, num_samples):
        """Draw collection information on the frame."""
        # Background rectangle for text
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Text information
        cv2.putText(frame, f"Gesture: {gesture_name}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {self.sample_count}/{num_samples}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Progress bar
        progress = self.sample_count / num_samples
        bar_width = 300
        bar_height = 20
        bar_x, bar_y = 20, 80
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        # Progress bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Instructions
        cv2.putText(frame, "Press 'c' to capture, 'q' to quit, 's' to skip", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _save_sample(self, landmarks, gesture_name):
        """Save a single sample of landmark data."""
        # Create sample data
        sample_data = {
            'gesture': gesture_name,
            'timestamp': datetime.now().isoformat(),
            'sample_id': f"{gesture_name}_{self.sample_count:03d}"
        }
        
        # Add landmark coordinates (21 landmarks x 2 coordinates = 42 features)
        for i, (x, y) in enumerate(landmarks):
            sample_data[f'landmark_{i}_x'] = x
            sample_data[f'landmark_{i}_y'] = y
        
        self.collected_data.append(sample_data)
    
    def save_collected_data(self, filename=None):
        """Save all collected data to CSV file."""
        if not self.collected_data:
            print("⚠️ No data to save.")
            return False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gesture_data_{timestamp}.csv"
        
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.collected_data)
        df.to_csv(filepath, index=False)
        
        print(f"💾 Data saved to: {filepath}")
        print(f"📊 Total samples: {len(self.collected_data)}")
        
        # Show gesture distribution
        gesture_counts = df['gesture'].value_counts()
        print("\n📈 Gesture distribution:")
        for gesture, count in gesture_counts.items():
            print(f"   {gesture}: {count} samples")
        
        return True
    
    def collect_all_gestures(self):
        """Collect data for all supported gestures."""
        print("🚀 Starting data collection for all gestures...")
        print(f"📋 Gestures to collect: {', '.join(SUPPORTED_GESTURES)}")
        
        if not self.initialize_camera():
            return False
        
        try:
            for gesture in SUPPORTED_GESTURES:
                print(f"\n{'='*50}")
                success = self.collect_gesture_data(gesture)
                if not success:
                    print(f"⚠️ Skipping {gesture} due to collection issues.")
                
                # Ask if user wants to continue
                print(f"\nPress 'y' to continue to next gesture, 'q' to quit, or 's' to save current data")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('y'):
                        break
                    elif key == ord('q'):
                        print("🛑 Collection stopped by user.")
                        return False
                    elif key == ord('s'):
                        self.save_collected_data()
                        return True
            
            # Save all data
            self.save_collected_data()
            print("\n🎉 All gesture data collection completed!")
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Collection interrupted by user.")
            return False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Resources cleaned up.")

def main():
    """Main function to run data collection."""
    print("🎯 Hand Helm - Data Collection Tool")
    print("=" * 40)
    
    collector = DataCollector()
    
    try:
        # Initialize camera
        if not collector.initialize_camera():
            return
        
        print("\nChoose collection mode:")
        print("1. Collect all gestures")
        print("2. Collect specific gesture")
        print("3. Test camera")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            collector.collect_all_gestures()
        elif choice == '2':
            print(f"\nAvailable gestures: {', '.join(SUPPORTED_GESTURES)}")
            gesture = input("Enter gesture name: ").strip()
            if gesture in SUPPORTED_GESTURES:
                collector.collect_gesture_data(gesture)
                collector.save_collected_data()
            else:
                print("❌ Invalid gesture name.")
        elif choice == '3':
            print("📹 Testing camera... Press 'q' to quit.")
            while True:
                success, frame = collector.cap.read()
                if not success:
                    break
                frame = cv2.flip(frame, 1)
                frame, _ = collector.detector.find_hands(frame)
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("❌ Invalid choice.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        collector.cleanup()

if __name__ == "__main__":
    main()
