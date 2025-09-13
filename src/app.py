#!/usr/bin/env python3
"""
Hand Helm - Main Application

This is the main application file that integrates all components:
- Hand detection and landmark extraction
- Gesture classification
- Computer control actions
- Real-time video feed with visual feedback

Usage:
    python src/app.py [options]
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from datetime import datetime
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, FPS, GESTURE_CONFIDENCE_THRESHOLD,
    GESTURE_HOLD_TIME, MAX_GESTURE_HISTORY, SUPPORTED_GESTURES
)
from src.gesture_detection import Hand_Detector
from src.gesture_classification import GestureClassifier
from src.computer_control import ComputerController, GestureActionManager

class HandHelmApp:
    """
    Main Hand Helm application class.
    
    This class orchestrates all components to provide real-time gesture recognition
    and computer control functionality.
    """
    
    def __init__(self, model_path=None, confidence_threshold=None, show_video=True):
        """
        Initialize the Hand Helm application.
        
        Args:
            model_path (str): Path to the trained model
            confidence_threshold (float): Minimum confidence for gesture recognition
            show_video (bool): Whether to show video feed
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold or GESTURE_CONFIDENCE_THRESHOLD
        self.show_video = show_video
        
        # Initialize components
        self.detector = None
        self.classifier = None
        self.controller = None
        self.action_manager = None
        
        # Gesture recognition state
        self.gesture_history = deque(maxlen=MAX_GESTURE_HISTORY)
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # seconds between predictions
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'actions_executed': 0,
            'start_time': time.time()
        }
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components."""
        print("🚀 Initializing Hand Helm...")
        
        try:
            # Initialize hand detector
            print("📹 Initializing hand detector...")
            self.detector = Hand_Detector()
            print("✅ Hand detector ready")
            
            # Initialize gesture classifier
            print("🧠 Loading gesture classifier...")
            self.classifier = GestureClassifier()
            if self.model_path and os.path.exists(self.model_path):
                self.classifier.load_model(self.model_path)
            else:
                self.classifier.load_model()  # Load default model
            print("✅ Gesture classifier ready")
            
            # Initialize computer controller
            print("🎮 Initializing computer controller...")
            self.controller = ComputerController()
            self.action_manager = GestureActionManager(self.controller)
            print("✅ Computer controller ready")
            
            print("🎉 Hand Helm initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
    
    def _process_gesture(self, landmarks):
        """
        Process hand landmarks to detect and classify gestures.
        
        Args:
            landmarks: Hand landmark data
        
        Returns:
            dict: Gesture prediction results
        """
        current_time = time.time()
        
        # Throttle predictions
        if current_time - self.last_prediction_time < self.prediction_interval:
            return None
        
        self.last_prediction_time = current_time
        
        try:
            # Convert landmarks to the format expected by classifier
            if landmarks and len(landmarks) > 0:
                # Take the first hand's landmarks
                hand_landmarks = landmarks[0]
                
                # Convert to numpy array (21 landmarks x 2 coordinates = 42 features)
                landmark_array = np.array(hand_landmarks).flatten()
                
                # Make prediction
                result = self.classifier.predict_gesture(
                    landmark_array, 
                    confidence_threshold=self.confidence_threshold
                )
                
                return result
                
        except Exception as e:
            print(f"⚠️ Error processing gesture: {e}")
        
        return None
    
    def _update_gesture_state(self, prediction):
        """
        Update the current gesture state based on prediction.
        
        Args:
            prediction: Gesture prediction results
        """
        if not prediction:
            return
        
        gesture = prediction['gesture']
        confidence = prediction['confidence']
        
        # Add to history
        self.gesture_history.append({
            'gesture': gesture,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Update current gesture
        if gesture != 'unknown' and confidence > self.confidence_threshold:
            if self.current_gesture == gesture:
                # Same gesture - check hold time
                if self.gesture_start_time:
                    hold_time = time.time() - self.gesture_start_time
                    if hold_time >= GESTURE_HOLD_TIME:
                        # Execute action
                        self._execute_gesture_action(gesture, confidence)
                        self.gesture_start_time = None  # Reset to prevent repeated actions
            else:
                # New gesture
                self.current_gesture = gesture
                self.gesture_start_time = time.time()
                print(f"🎯 New gesture detected: {gesture} (confidence: {confidence:.3f})")
        else:
            # No valid gesture
            if self.current_gesture:
                print(f"🔄 Gesture lost: {self.current_gesture}")
            self.current_gesture = None
            self.gesture_start_time = None
    
    def _execute_gesture_action(self, gesture, confidence):
        """
        Execute the action associated with a gesture.
        
        Args:
            gesture (str): The detected gesture
            confidence (float): Confidence level
        """
        success = self.action_manager.process_gesture(gesture, confidence)
        
        if success:
            self.stats['actions_executed'] += 1
            print(f"🎯 Action executed: {gesture} (confidence: {confidence:.3f})")
        else:
            print(f"⚠️ Failed to execute action for gesture: {gesture}")
    
    def _draw_interface(self, frame, prediction=None):
        """
        Draw the user interface on the video frame.
        
        Args:
            frame: Video frame
            prediction: Current gesture prediction
        """
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Hand Helm", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Current gesture
        if prediction and prediction['gesture'] != 'unknown':
            gesture = prediction['gesture']
            confidence = prediction['confidence']
            
            # Gesture name with better visibility
            color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 255, 255)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Confidence bar
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 20, 90
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (100, 100, 100), -1)
            
            # Confidence bar
            conf_width = int(bar_width * confidence)
            color = (0, 255, 0) if confidence > self.confidence_threshold else (0, 255, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                         color, -1)
            
            # Confidence text
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Hold time indicator
            if self.gesture_start_time:
                hold_time = time.time() - self.gesture_start_time
                progress = min(hold_time / GESTURE_HOLD_TIME, 1.0)
                
                hold_bar_width = int(bar_width * progress)
                cv2.rectangle(frame, (bar_x, bar_y + 30), (bar_x + hold_bar_width, bar_y + 50), 
                             (255, 255, 0), -1)
                
                cv2.putText(frame, f"Hold: {hold_time:.1f}s", (20, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No gesture detected", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Statistics
        stats_text = f"Frames: {self.stats['frames_processed']} | Actions: {self.stats['actions_executed']}"
        cv2.putText(frame, stats_text, (20, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to show/hide stats",
            "Press 'r' to reset stats"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 200, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Run the main application loop."""
        print("🎬 Starting Hand Helm...")
        print("=" * 40)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        print("📹 Camera initialized")
        print("🎯 Gesture recognition active")
        print("🎮 Computer control enabled")
        print("\nPress 'q' to quit, 's' for stats, 'r' to reset")
        
        try:
            while True:
                # Read frame
                success, frame = cap.read()
                if not success:
                    print("❌ Error: Failed to capture frame.")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect hands and extract landmarks
                frame, landmarks = self.detector.find_hands(frame)
                
                # Process gesture if hands detected
                prediction = None
                if landmarks:
                    prediction = self._process_gesture(landmarks)
                    if prediction:
                        self.stats['gestures_detected'] += 1
                        self._update_gesture_state(prediction)
                
                # Draw interface
                if self.show_video:
                    self._draw_interface(frame, prediction)
                    cv2.imshow("Hand Helm - Gesture Control", frame)
                
                # Update statistics
                self.stats['frames_processed'] += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    self._show_statistics()
                elif key == ord('r'):
                    self._reset_statistics()
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self._show_final_statistics()
    
    def _show_statistics(self):
        """Show current statistics."""
        runtime = time.time() - self.stats['start_time']
        fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
        
        print(f"\n📊 Hand Helm Statistics:")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   Frames processed: {self.stats['frames_processed']}")
        print(f"   Gestures detected: {self.stats['gestures_detected']}")
        print(f"   Actions executed: {self.stats['actions_executed']}")
        
        # Action manager statistics
        if self.action_manager:
            manager_stats = self.action_manager.get_statistics()
            print(f"   Success rate: {manager_stats['success_rate']:.2%}")
    
    def _reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'actions_executed': 0,
            'start_time': time.time()
        }
        if self.action_manager:
            self.action_manager.clear_history()
        print("🔄 Statistics reset")
    
    def _show_final_statistics(self):
        """Show final statistics on exit."""
        print("\n🎉 Hand Helm Session Complete!")
        self._show_statistics()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Hand Helm - Gesture Control Application")
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=GESTURE_CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for gesture recognition')
    parser.add_argument('--no-video', action='store_true', help='Run without video display')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    print("🎯 Hand Helm - Gesture Control System")
    print("=" * 50)
    
    try:
        # Create and run application
        app = HandHelmApp(
            model_path=args.model,
            confidence_threshold=args.confidence,
            show_video=not args.no_video
        )
        
        if args.test:
            print("🧪 Running in test mode...")
            # Run a quick test
            app._show_statistics()
        else:
            app.run()
    
    except Exception as e:
        print(f"❌ Error running application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
