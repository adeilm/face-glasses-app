#!/usr/bin/env python3
"""
Test script for MediaPipe-based 3D glasses overlay system
"""

import cv2
import os
import sys
import traceback
import numpy as np

def test_mediapipe_system():
    """Test the MediaPipe-based 3D glasses system"""
    print("=== MediaPipe 3D Glasses System Test ===\n")
    
    # Test imports
    print("1. Testing imports...")
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import trimesh
        print("✅ Trimesh imported successfully")
    except ImportError as e:
        print(f"❌ Trimesh import failed: {e}")
        return False
    
    try:
        from mediapipe_3d_glasses import MediaPipe3DGlasses
        print("✅ MediaPipe3DGlasses imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe3DGlasses import failed: {e}")
        traceback.print_exc()
        return False
    
    # Test GLB file
    print("\n2. Testing GLB file...")
    glb_path = "assets/glasses/glasses.glb"
    if not os.path.exists(glb_path):
        print(f"❌ GLB file not found: {glb_path}")
        return False
    print(f"✅ GLB file found: {glb_path}")
    
    # Test image file
    print("\n3. Testing input image...")
    test_image_path = "uploads/carmen-electra-face-profile-earrings-wallpaper-preview.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    print(f"✅ Test image found: {test_image_path}")
    
    # Initialize system
    print("\n4. Initializing MediaPipe 3D Glasses system...")
    try:
        glasses_system = MediaPipe3DGlasses(glb_path, show_debug=True)
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        traceback.print_exc()
        return False
    
    # Test static image processing
    print("\n5. Testing static image processing...")
    try:
        result = glasses_system.process_image(test_image_path, "test_result_mediapipe.jpg")
        if result is not None:
            print("✅ Static image processing successful")
            print("📁 Result saved as: test_result_mediapipe.jpg")
        else:
            print("❌ Static image processing failed")
            return False
    except Exception as e:
        print(f"❌ Static image processing failed: {e}")
        traceback.print_exc()
        return False
    
    # Test face detection
    print("\n6. Testing face detection...")
    try:
        image = cv2.imread(test_image_path)
        landmarks = glasses_system.extract_face_landmarks(image)
        if landmarks is not None:
            print(f"✅ Face detection successful - found {len(landmarks)} landmarks")
        else:
            print("❌ No face detected")
            return False
    except Exception as e:
        print(f"❌ Face detection failed: {e}")
        traceback.print_exc()
        return False
    
    # Test head pose estimation
    print("\n7. Testing head pose estimation...")
    try:
        rotation_vector, translation_vector = glasses_system.estimate_head_pose(landmarks, image.shape[:2])
        if rotation_vector is not None and translation_vector is not None:
            print("✅ Head pose estimation successful")
            print(f"   Rotation vector: {rotation_vector.flatten()}")
            print(f"   Translation vector: {translation_vector.flatten()}")
        else:
            print("❌ Head pose estimation failed")
            return False
    except Exception as e:
        print(f"❌ Head pose estimation failed: {e}")
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\n8. Cleaning up...")
    try:
        glasses_system.cleanup()
        print("✅ Cleanup successful")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        traceback.print_exc()
    
    print("\n=== All tests completed successfully! ===")
    print("\nTo run the full system:")
    print("1. For static image: python mediapipe_3d_glasses.py")
    print("2. For webcam: python mediapipe_3d_glasses.py (choose option 2)")
    
    return True

def quick_webcam_test():
    """Quick test of webcam functionality"""
    print("\n=== Quick Webcam Test ===")
    
    try:
        from mediapipe_3d_glasses import MediaPipe3DGlasses
        
        glb_path = "assets/glasses/glasses.glb"
        if not os.path.exists(glb_path):
            print(f"❌ GLB file not found: {glb_path}")
            return False
        
        print("Initializing webcam test...")
        glasses_system = MediaPipe3DGlasses(glb_path, show_debug=True)
        
        # Test webcam access
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access webcam")
            return False
        
        print("✅ Webcam access successful")
        print("Press 'q' to quit this test")
        
        # Process a few frames
        for i in range(100):  # Test 100 frames
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame = glasses_system.process_frame(frame)
            
            cv2.imshow('MediaPipe 3D Glasses Test', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        glasses_system.cleanup()
        
        print("✅ Webcam test completed")
        return True
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("MediaPipe 3D Glasses System Test Suite")
    print("=====================================")
    
    # Run comprehensive tests
    if not test_mediapipe_system():
        print("\n❌ System tests failed!")
        return
    
    # Ask for webcam test
    while True:
        choice = input("\nWould you like to run a quick webcam test? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            quick_webcam_test()
            break
        elif choice in ['n', 'no']:
            print("Skipping webcam test")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
