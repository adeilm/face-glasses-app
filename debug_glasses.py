#!/usr/bin/env python3
"""
Debug script for MediaPipe 3D glasses rendering
"""

import cv2
import numpy as np
import sys
import os
import traceback

def debug_glasses_rendering():
    """Debug the glasses rendering process step by step"""
    print("=== MediaPipe 3D Glasses Debug ===")
    
    try:
        # Import the system
        from mediapipe_3d_glasses import MediaPipe3DGlasses
        
        # Initialize system
        glb_path = "assets/glasses/glasses.glb"
        glasses_system = MediaPipe3DGlasses(glb_path, show_debug=True)
        
        # Analyze the model
        glasses_system.analyze_glasses_model()
        
        # Load test image
        test_image = "uploads/carmen-electra-face-profile-earrings-wallpaper-preview.jpg"
        image = cv2.imread(test_image)
        
        if image is None:
            print(f"❌ Could not load test image: {test_image}")
            return
        
        print(f"✅ Loaded test image: {image.shape}")
        
        # Extract landmarks
        print("\n=== Face Detection ===")
        landmarks_3d = glasses_system.extract_face_landmarks(image)
        
        if landmarks_3d is None:
            print("❌ No face detected")
            return
        
        print(f"✅ Detected {len(landmarks_3d)} facial landmarks")
        
        # Print key landmark positions
        print("\n=== Key Landmarks ===")
        for name, idx in glasses_system.glasses_landmarks.items():
            pos = landmarks_3d[idx]
            print(f"{name}: {pos}")
        
        # Estimate head pose
        print("\n=== Head Pose Estimation ===")
        rotation_vector, translation_vector = glasses_system.estimate_head_pose(landmarks_3d, image.shape[:2])
        
        if rotation_vector is None:
            print("❌ Head pose estimation failed")
            return
        
        print(f"✅ Head pose estimated")
        print(f"Rotation vector: {rotation_vector.flatten()}")
        print(f"Translation vector: {translation_vector.flatten()}")
        
        # Calculate glasses parameters
        print("\n=== Glasses Positioning ===")
        glasses_position, glasses_rotation, scale_factor = glasses_system.calculate_glasses_transform(
            landmarks_3d, rotation_vector, translation_vector
        )
        
        print(f"Glasses position: {glasses_position}")
        print(f"Glasses rotation: {glasses_rotation.flatten()}")
        print(f"Scale factor: {scale_factor}")
        
        # Render glasses step by step
        print("\n=== Rendering Process ===")
        
        # Create a copy for visualization
        debug_image = image.copy()
        
        # Draw facial landmarks
        for idx in glasses_system.glasses_landmarks.values():
            pos = landmarks_3d[idx][:2].astype(int)
            cv2.circle(debug_image, tuple(pos), 3, (0, 255, 0), -1)
        
        # Draw glasses center
        cv2.circle(debug_image, tuple(glasses_position[:2].astype(int)), 8, (0, 255, 255), -1)
        
        # Process the full frame
        result_image = glasses_system.process_frame(image)
        
        # Save debug images
        cv2.imwrite("debug_landmarks.jpg", debug_image)
        cv2.imwrite("debug_glasses_result.jpg", result_image)
        
        print("✅ Debug images saved:")
        print("  - debug_landmarks.jpg (landmarks visualization)")
        print("  - debug_glasses_result.jpg (final result)")
        
        # Show results
        cv2.imshow("Debug - Landmarks", debug_image)
        cv2.imshow("Debug - Final Result", result_image)
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Cleanup
        glasses_system.cleanup()
        
        print("\n✅ Debug completed successfully!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_glasses_rendering()
