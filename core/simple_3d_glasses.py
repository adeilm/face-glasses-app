#!/usr/bin/env python3
"""
Simplified 3D Glasses Overlay with Better Visibility
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import os
import logging
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMediaPipe3DGlasses:
    """
    Simplified MediaPipe 3D Glasses with better visibility
    """
    
    def __init__(self, show_debug: bool = False):
        self.show_debug = show_debug
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Key landmarks for glasses
        self.glasses_landmarks = {
            'left_eye_outer': 33,
            'left_eye_inner': 133,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'nose_bridge': 168,
            'left_eyebrow_outer': 70,
            'right_eyebrow_outer': 300,
        }
        
        logger.info("Simple MediaPipe 3D Glasses initialized")
    
    def initialize_camera_params(self, width: int, height: int):
        """Initialize camera parameters"""
        focal_length = width
        center = (width / 2, height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def extract_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        height, width = image.shape[:2]
        landmarks_3d = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            landmarks_3d.append([x, y, z])
        
        return np.array(landmarks_3d, dtype=np.float64)
    
    def draw_3d_glasses(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw 3D-looking glasses using OpenCV primitives"""
        if landmarks is None:
            return image
        
        try:
            # Get key points
            left_eye_outer = landmarks[self.glasses_landmarks['left_eye_outer']][:2].astype(int)
            left_eye_inner = landmarks[self.glasses_landmarks['left_eye_inner']][:2].astype(int)
            right_eye_inner = landmarks[self.glasses_landmarks['right_eye_inner']][:2].astype(int)
            right_eye_outer = landmarks[self.glasses_landmarks['right_eye_outer']][:2].astype(int)
            nose_bridge = landmarks[self.glasses_landmarks['nose_bridge']][:2].astype(int)
            left_eyebrow_outer = landmarks[self.glasses_landmarks['left_eyebrow_outer']][:2].astype(int)
            right_eyebrow_outer = landmarks[self.glasses_landmarks['right_eyebrow_outer']][:2].astype(int)
            
            # Calculate eye centers and sizes
            left_eye_center = ((left_eye_outer + left_eye_inner) / 2).astype(int)
            right_eye_center = ((right_eye_outer + right_eye_inner) / 2).astype(int)
            
            # Calculate lens sizes
            left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
            right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
            
            # Average eye dimensions
            avg_eye_width = int((left_eye_width + right_eye_width) / 2)
            lens_radius = int(avg_eye_width * 0.8)
            
            # Colors
            frame_color = (0, 0, 0)      # Black frames
            lens_color = (200, 200, 200)  # Light gray lenses
            highlight_color = (255, 255, 255)  # White highlights
            
            # Draw left lens
            # Outer circle (frame)
            cv2.circle(image, tuple(left_eye_center), lens_radius + 3, frame_color, 4)
            # Inner circle (lens) - semi-transparent
            overlay = image.copy()
            cv2.circle(overlay, tuple(left_eye_center), lens_radius, lens_color, -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            # Highlight
            highlight_pos = (left_eye_center[0] - lens_radius//3, left_eye_center[1] - lens_radius//3)
            cv2.circle(image, highlight_pos, lens_radius//4, highlight_color, -1)
            
            # Draw right lens
            # Outer circle (frame)
            cv2.circle(image, tuple(right_eye_center), lens_radius + 3, frame_color, 4)
            # Inner circle (lens) - semi-transparent
            overlay = image.copy()
            cv2.circle(overlay, tuple(right_eye_center), lens_radius, lens_color, -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            # Highlight
            highlight_pos = (right_eye_center[0] - lens_radius//3, right_eye_center[1] - lens_radius//3)
            cv2.circle(image, highlight_pos, lens_radius//4, highlight_color, -1)
            
            # Draw nose bridge
            bridge_start = (left_eye_center[0] + lens_radius//2, left_eye_center[1])
            bridge_end = (right_eye_center[0] - lens_radius//2, right_eye_center[1])
            cv2.line(image, bridge_start, bridge_end, frame_color, 4)
            
            # Draw left temple
            temple_start = (left_eye_center[0] - lens_radius - 3, left_eye_center[1])
            temple_end = (left_eyebrow_outer[0] - 20, left_eyebrow_outer[1])
            cv2.line(image, temple_start, temple_end, frame_color, 4)
            
            # Draw right temple
            temple_start = (right_eye_center[0] + lens_radius + 3, right_eye_center[1])
            temple_end = (right_eyebrow_outer[0] + 20, right_eyebrow_outer[1])
            cv2.line(image, temple_start, temple_end, frame_color, 4)
            
            # Add 3D effect with shadows
            shadow_offset = 3
            shadow_color = (50, 50, 50)
            
            # Shadow for left lens
            shadow_center = (left_eye_center[0] + shadow_offset, left_eye_center[1] + shadow_offset)
            cv2.circle(image, shadow_center, lens_radius + 3, shadow_color, 2)
            
            # Shadow for right lens
            shadow_center = (right_eye_center[0] + shadow_offset, right_eye_center[1] + shadow_offset)
            cv2.circle(image, shadow_center, lens_radius + 3, shadow_color, 2)
            
            if self.show_debug:
                # Draw landmark points
                cv2.circle(image, tuple(left_eye_center), 3, (0, 255, 0), -1)
                cv2.circle(image, tuple(right_eye_center), 3, (0, 255, 0), -1)
                cv2.circle(image, tuple(nose_bridge), 3, (0, 255, 255), -1)
                
                # Display eye measurements
                cv2.putText(image, f"Eye width: {avg_eye_width:.1f}px", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Lens radius: {lens_radius}px", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image
            
        except Exception as e:
            logger.error(f"Error drawing glasses: {e}")
            return image
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        # Initialize camera parameters if needed
        if self.camera_matrix is None:
            self.initialize_camera_params(frame.shape[1], frame.shape[0])
        
        # Extract landmarks
        landmarks = self.extract_face_landmarks(frame)
        
        if landmarks is None:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Draw glasses
        frame = self.draw_3d_glasses(frame, landmarks)
        
        return frame
    
    def process_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """Process a static image"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        result = self.process_frame(image)
        
        if output_path:
            cv2.imwrite(output_path, result)
            logger.info(f"Output saved to: {output_path}")
        
        return result
    
    def run_webcam(self, camera_id: int = 0):
        """Run real-time webcam processing"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info("Starting webcam. Press 'q' to quit, 'd' to toggle debug mode.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('Simple 3D Glasses', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                logger.info(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

def main():
    """Main function"""
    print("Choose processing mode:")
    print("1. Static image processing")
    print("2. Real-time webcam processing")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    # Initialize system
    glasses_system = SimpleMediaPipe3DGlasses(show_debug=True)
    
    try:
        if choice == "1":
            image_path = input("Enter path to input image: ").strip()
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                return
            
            output_path = "simple_glasses_result.jpg"
            result = glasses_system.process_image(image_path, output_path)
            
            if result is not None:
                cv2.imshow("Result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        elif choice == "2":
            camera_id = int(input("Enter camera ID (0 for default): ").strip() or "0")
            glasses_system.run_webcam(camera_id)
        
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        glasses_system.cleanup()

if __name__ == "__main__":
    main()
