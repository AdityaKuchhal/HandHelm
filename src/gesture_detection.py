import cv2
import mediapipe as mp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE)

class Hand_Detector:
    def __init__(self, 
                 max_num_hands = MAX_NUM_HANDS, 
                 min_detection_confidence = MIN_DETECTION_CONFIDENCE, 
                 min_tracking_confidence = MIN_TRACKING_CONFIDENCE): 
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence            
        )
    
    def find_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        landmark_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))
                landmark_data.append(landmarks)
                
        return frame, landmark_data
    
    def __del__(self):
        self.hands.close()

def main():
    detector = Hand_Detector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        success, frame = cap.read()
        
        if not success:
            print("Error: Failed to capture frame.")
            break
        
        frame = cv2.flip(frame, 1)
        frame, landmark_data = detector.find_hands(frame)
        
        cv2.imshow("Hand Helm - Hand Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()