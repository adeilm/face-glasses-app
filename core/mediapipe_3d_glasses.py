#!/usr/bin/env python3
"""
Advanced 3D Glasses Overlay using MediaPipe Face Mesh
=====================================================

This script implements a real-time 3D glasses overlay system using:
- MediaPipe Face Mesh for 468 3D facial landmarks
- OpenCV solvePnP for head pose estimation
- Trimesh for 3D model handling
- Real-time webcam processing or static image processing

Author: AI Assistant
Date: July 2025
"""

import cv2
import numpy as np
import mediapipe as mp
import trimesh
import math
import os
import sys
import time
from typing import List, Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipe3DGlasses:
    """
    Advanced 3D Glasses Overlay using MediaPipe Face Mesh
    """
    
    def __init__(self, glb_path: str, show_debug: bool = False):
        """
        Initialize the MediaPipe 3D Glasses system
        
        Args:
            glb_path: Path to the .glb glasses model file
            show_debug: Whether to show debug information (pose axes, landmarks)
        """
        self.glb_path = glb_path
        self.show_debug = show_debug
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Face Mesh with high-quality settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load 3D glasses model
        self.glasses_model = self.load_glasses_model()
        
        # Define 3D model points for head pose estimation (in mm)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera parameters (will be updated based on frame size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # MediaPipe landmark indices for pose estimation
        self.pose_landmark_indices = [
            1,    # Nose tip
            152,  # Chin
            33,   # Left eye left corner
            362,  # Right eye right corner
            61,   # Left mouth corner
            291   # Right mouth corner
        ]
        
        # Glasses positioning landmarks (eye corners and bridge)
        self.glasses_landmarks = {
            'left_eye_outer': 33,
            'left_eye_inner': 133,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'nose_bridge': 168,
            'nose_tip': 1
        }
        
        logger.info("MediaPipe 3D Glasses system initialized")
    
    def load_glasses_model(self) -> Optional[trimesh.Trimesh]:
        """
        Load the 3D glasses model from GLB file
        
        Returns:
            Trimesh object or None if loading fails
        """
        try:
            if not os.path.exists(self.glb_path):
                logger.error(f"GLB file not found: {self.glb_path}")
                return None
            
            # Load the GLB file
            scene = trimesh.load(self.glb_path)
            
            # If it's a scene with multiple meshes, combine them
            if isinstance(scene, trimesh.Scene):
                # Combine all meshes in the scene
                combined_mesh = trimesh.util.concatenate([
                    mesh for mesh in scene.geometry.values()
                    if isinstance(mesh, trimesh.Trimesh)
                ])
                glasses_model = combined_mesh
            else:
                glasses_model = scene
            
            # Normalize the model size
            glasses_model.apply_scale(0.1)  # Adjust scale as needed
            
            logger.info(f"Successfully loaded glasses model: {self.glb_path}")
            logger.info(f"Model bounds: {glasses_model.bounds}")
            
            return glasses_model
            
        except Exception as e:
            logger.error(f"Error loading glasses model: {e}")
            return None
    
    def initialize_camera_params(self, frame_width: int, frame_height: int):
        """
        Initialize camera parameters based on frame size
        
        Args:
            frame_width: Width of the video frame
            frame_height: Height of the video frame
        """
        # Estimate camera matrix (assuming typical webcam FOV)
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        logger.info(f"Camera parameters initialized for {frame_width}x{frame_height}")
    
    def extract_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 3D facial landmarks using MediaPipe Face Mesh
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Array of 468 3D landmarks or None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array with 3D coordinates
        height, width = image.shape[:2]
        landmarks_3d = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # Relative depth
            landmarks_3d.append([x, y, z])
        
        return np.array(landmarks_3d, dtype=np.float64)
    
    def estimate_head_pose(self, landmarks_3d: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate head pose using OpenCV solvePnP
        
        Args:
            landmarks_3d: 3D facial landmarks
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of (rotation_vector, translation_vector)
        """
        if self.camera_matrix is None:
            self.initialize_camera_params(image_shape[1], image_shape[0])
        
        # Extract 2D points for pose estimation
        image_points = np.array([
            landmarks_3d[idx][:2] for idx in self.pose_landmark_indices
        ], dtype=np.float64)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            logger.warning("PnP solver failed")
            return None, None
        
        return rotation_vector, translation_vector
    
    def calculate_glasses_transform(self, landmarks_3d: np.ndarray, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate transformation parameters for glasses positioning
        
        Args:
            landmarks_3d: 3D facial landmarks
            rotation_vector: Head rotation vector
            translation_vector: Head translation vector
            
        Returns:
            Tuple of (glasses_position, glasses_rotation, scale_factor)
        """
        # Get key points for glasses positioning
        left_eye_outer = landmarks_3d[self.glasses_landmarks['left_eye_outer']][:2]
        right_eye_outer = landmarks_3d[self.glasses_landmarks['right_eye_outer']][:2]
        left_eye_inner = landmarks_3d[self.glasses_landmarks['left_eye_inner']][:2]
        right_eye_inner = landmarks_3d[self.glasses_landmarks['right_eye_inner']][:2]
        nose_bridge = landmarks_3d[self.glasses_landmarks['nose_bridge']][:2]
        
        # Calculate glasses center (between eye centers)
        left_eye_center = (left_eye_outer + left_eye_inner) / 2
        right_eye_center = (right_eye_outer + right_eye_inner) / 2
        glasses_center = (left_eye_center + right_eye_center) / 2
        
        # Calculate inter-eye distance for scaling
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        scale_factor = eye_distance / 60.0  # Adjusted scale for better fit
        
        # Adjust glasses position (slightly above the eye line)
        glasses_position = np.array([glasses_center[0], glasses_center[1] - 10, 0])
        
        return glasses_position, rotation_vector, scale_factor
    
    def render_glasses_on_face(self, image: np.ndarray, landmarks_3d: np.ndarray, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """
        Render 3D glasses on the face
        
        Args:
            image: Input image
            landmarks_3d: 3D facial landmarks
            rotation_vector: Head rotation vector
            translation_vector: Head translation vector
            
        Returns:
            Image with glasses overlay
        """
        if self.glasses_model is None:
            logger.warning("No glasses model loaded")
            return image
        
        try:
            # Calculate transformation parameters
            glasses_position, glasses_rotation, scale_factor = self.calculate_glasses_transform(
                landmarks_3d, rotation_vector, translation_vector
            )
            
            # Get scaled and rotated vertices
            glasses_vertices = self.glasses_model.vertices.copy()
            
            # Apply scaling
            glasses_vertices *= scale_factor
            
            # Apply rotation
            rotation_matrix, _ = cv2.Rodrigues(glasses_rotation)
            glasses_vertices = glasses_vertices @ rotation_matrix.T
            
            # Create 3D points relative to glasses position
            glasses_3d_points = glasses_vertices + [0, 0, -50]  # Move glasses forward
            
            # Project 3D points to 2D using camera parameters
            projected_points, _ = cv2.projectPoints(
                glasses_3d_points.reshape(-1, 1, 3),
                np.zeros(3),  # No additional rotation
                glasses_position.reshape(3, 1),  # Translation
                self.camera_matrix,
                self.dist_coeffs
            )
            
            projected_points = projected_points.reshape(-1, 2).astype(np.int32)
            
            # Ensure points are within image bounds
            height, width = image.shape[:2]
            valid_indices = []
            for i, point in enumerate(projected_points):
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    valid_indices.append(i)
            
            # Draw glasses with solid fill and wireframe
            for face in self.glasses_model.faces:
                if len(face) == 3:  # Triangle
                    # Check if all vertices are valid
                    if all(idx in valid_indices for idx in face):
                        pts = projected_points[face]
                        
                        # Draw filled triangle (semi-transparent)
                        overlay = image.copy()
                        cv2.fillPoly(overlay, [pts], (100, 100, 100))
                        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                        
                        # Draw wireframe
                        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            
            # Draw additional visual elements for better visibility
            # Draw eye positions
            left_eye_center = (landmarks_3d[self.glasses_landmarks['left_eye_inner']][:2] + 
                              landmarks_3d[self.glasses_landmarks['left_eye_outer']][:2]) / 2
            right_eye_center = (landmarks_3d[self.glasses_landmarks['right_eye_inner']][:2] + 
                               landmarks_3d[self.glasses_landmarks['right_eye_outer']][:2]) / 2
            
            cv2.circle(image, tuple(left_eye_center.astype(int)), 3, (255, 0, 0), -1)
            cv2.circle(image, tuple(right_eye_center.astype(int)), 3, (255, 0, 0), -1)
            
            # Draw glasses center
            cv2.circle(image, tuple(glasses_position[:2].astype(int)), 5, (0, 255, 255), -1)
            
            return image
            
        except Exception as e:
            logger.error(f"Error rendering glasses: {e}")
            import traceback
            traceback.print_exc()
            return image
    
    def draw_debug_info(self, image: np.ndarray, landmarks_3d: np.ndarray, rotation_vector: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """
        Draw debug information (pose axes, landmarks)
        
        Args:
            image: Input image
            landmarks_3d: 3D facial landmarks
            rotation_vector: Head rotation vector
            translation_vector: Head translation vector
            
        Returns:
            Image with debug info
        """
        if not self.show_debug:
            return image
        
        # Draw pose axes
        axis_points = np.array([
            [0, 0, 0],      # Origin
            [100, 0, 0],    # X-axis (red)
            [0, 100, 0],    # Y-axis (green)
            [0, 0, -100]    # Z-axis (blue)
        ], dtype=np.float64)
        
        projected_axes, _ = cv2.projectPoints(
            axis_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        projected_axes = projected_axes.reshape(-1, 2).astype(np.int32)
        
        # Draw axes
        origin = tuple(projected_axes[0])
        cv2.arrowedLine(image, origin, tuple(projected_axes[1]), (0, 0, 255), 3)  # X-axis (red)
        cv2.arrowedLine(image, origin, tuple(projected_axes[2]), (0, 255, 0), 3)  # Y-axis (green)
        cv2.arrowedLine(image, origin, tuple(projected_axes[3]), (255, 0, 0), 3)  # Z-axis (blue)
        
        # Draw key landmarks
        for idx in self.pose_landmark_indices:
            point = tuple(landmarks_3d[idx][:2].astype(int))
            cv2.circle(image, point, 3, (255, 255, 0), -1)
        
        # Calculate and display angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)))
        roll = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        
        # Display pose information
        cv2.putText(image, f"Pitch: {pitch:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Yaw: {yaw:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Roll: {roll:.1f}°", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for 3D glasses overlay
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Processed frame with glasses overlay
        """
        # Extract facial landmarks
        landmarks_3d = self.extract_face_landmarks(frame)
        
        if landmarks_3d is None:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Estimate head pose
        rotation_vector, translation_vector = self.estimate_head_pose(landmarks_3d, frame.shape[:2])
        
        if rotation_vector is None or translation_vector is None:
            cv2.putText(frame, "Pose estimation failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Render glasses
        frame = self.render_glasses_on_face(frame, landmarks_3d, rotation_vector, translation_vector)
        
        # Draw debug info if enabled
        frame = self.draw_debug_info(frame, landmarks_3d, rotation_vector, translation_vector)
        
        return frame
    
    def process_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Process a static image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            
        Returns:
            Processed image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Process frame
        result = self.process_frame(image)
        
        # Save output if path provided
        if output_path:
            cv2.imwrite(output_path, result)
            logger.info(f"Output saved to: {output_path}")
        
        return result
    
    def run_webcam(self, camera_id: int = 0):
        """
        Run real-time webcam processing
        
        Args:
            camera_id: Camera device ID (default: 0)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting webcam processing. Press 'q' to quit, 'd' to toggle debug mode.")
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = fps_counter / elapsed_time
                logger.info(f"FPS: {fps:.1f}")
            
            # Display frame
            cv2.imshow('MediaPipe 3D Glasses Overlay', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                logger.info(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam processing stopped")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("Resources cleaned up")
    
    def analyze_glasses_model(self):
        """
        Analyze and print information about the loaded glasses model
        """
        if self.glasses_model is None:
            logger.error("No glasses model loaded")
            return
        
        try:
            print(f"=== Glasses Model Analysis ===")
            print(f"Model type: {type(self.glasses_model)}")
            print(f"Number of vertices: {len(self.glasses_model.vertices)}")
            print(f"Number of faces: {len(self.glasses_model.faces)}")
            print(f"Model bounds: {self.glasses_model.bounds}")
            print(f"Model center: {self.glasses_model.center_mass}")
            print(f"Model scale: {self.glasses_model.scale}")
            
            # Print some sample vertices
            print(f"Sample vertices (first 5):")
            for i in range(min(5, len(self.glasses_model.vertices))):
                print(f"  Vertex {i}: {self.glasses_model.vertices[i]}")
            
            # Check if model has materials or textures
            if hasattr(self.glasses_model, 'visual'):
                print(f"Model has visual properties: {self.glasses_model.visual}")
            
            print(f"==============================")
            
        except Exception as e:
            logger.error(f"Error analyzing glasses model: {e}")
            import traceback
            traceback.print_exc()
    
def main():
    """Main function for testing"""
    # Configuration
    GLB_PATH = "assets/glasses/glasses.glb"
    SHOW_DEBUG = True
    
    # Check if GLB file exists
    if not os.path.exists(GLB_PATH):
        logger.error(f"GLB file not found: {GLB_PATH}")
        logger.info("Please place your glasses.glb file in the assets/glasses/ folder")
        return
    
    # Initialize the system
    glasses_system = MediaPipe3DGlasses(GLB_PATH, show_debug=SHOW_DEBUG)
    
    try:
        # Test modes
        print("\nChoose processing mode:")
        print("1. Static image processing")
        print("2. Real-time webcam processing")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            # Static image processing
            image_path = input("Enter path to input image: ").strip()
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return
            
            output_path = "output_mediapipe_3d_glasses.jpg"
            result = glasses_system.process_image(image_path, output_path)
            
            if result is not None:
                cv2.imshow("Result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        elif choice == "2":
            # Real-time webcam processing
            camera_id = int(input("Enter camera ID (0 for default): ").strip() or "0")
            glasses_system.run_webcam(camera_id)
        
        else:
            logger.error("Invalid choice")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        glasses_system.cleanup()

if __name__ == "__main__":
    main()
