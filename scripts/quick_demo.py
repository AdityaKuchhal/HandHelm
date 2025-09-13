#!/usr/bin/env python3
"""
Quick demo script to test HandHelm with improved settings.
"""

import os
import sys
import cv2
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gesture_detection import Hand_Detector
from src.gesture_classification import GestureClassifier
from src.computer_control import ComputerController, GestureActionManager
from src.config import GESTURE_CONFIDENCE_THRESHOLD

def quick_demo():
    """Run a quick demo of the HandHelm system."""
    print("🎯 Hand Helm - Quick Demo")
    print("=" * 40)
    print("This demo will test gesture detection and control actions.")
    print("Make sure your webcam is working and you have good lighting!")
    print("\n🎮 Controls:")
    print("   Press 'q' or ESC to quit")
    print("   Press 's' to test actions manually")
    print("   Press Ctrl+C in terminal to force quit")
    
    # Initialize components
    print("\n🚀 Initializing components...")
    
    detector = Hand_Detector()
    classifier = GestureClassifier()
    classifier.load_model()
    controller = ComputerController()
    action_manager = GestureActionManager(controller)
    
    print("✅ All components ready!")
    print(f"   Confidence threshold: {GESTURE_CONFIDENCE_THRESHOLD}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return
    
    print("\n📹 Camera opened - start making gestures!")
    print("🎮 Try: fist, open_palm, thumbs_up, thumbs_down")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            frame, landmarks = detector.find_hands(frame)
            
            # Process gesture if hands detected
            if landmarks and len(landmarks) > 0:
                # Convert landmarks
                hand_landmarks = landmarks[0]
                landmark_array = np.array(hand_landmarks).flatten()
                
                # Make prediction
                result = classifier.predict_gesture(landmark_array)
                
                if result and result['gesture'] != 'unknown':
                    gesture = result['gesture']
                    confidence = result['confidence']
                    
                    # Display gesture info
                    cv2.putText(frame, f"Gesture: {gesture}", (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Test action execution
                    if confidence > GESTURE_CONFIDENCE_THRESHOLD:
                        cv2.putText(frame, "EXECUTING ACTION!", (20, 110), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Execute action
                        success = action_manager.process_gesture(gesture, confidence)
                        if success:
                            cv2.putText(frame, "ACTION EXECUTED!", (20, 140), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "ACTION FAILED!", (20, 140), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add instructions to frame
            cv2.putText(frame, "Press 'q' or ESC to quit", (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 's' to test actions", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Hand Helm - Quick Demo", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC key
                print("\n🛑 Quitting demo...")
                break
            elif key == ord('s'):
                # Test actions manually
                print("\n🧪 Testing actions manually...")
                test_gestures = ['fist', 'open_palm', 'thumbs_up', 'thumbs_down']
                for gesture in test_gestures:
                    success = action_manager.process_gesture(gesture, 0.8)
                    print(f"   {gesture}: {'✅' if success else '❌'}")
                    time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n🎉 Demo completed!")

if __name__ == "__main__":
    import numpy as np
    quick_demo()
