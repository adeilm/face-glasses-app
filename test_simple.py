#!/usr/bin/env python3

import cv2
import os
import sys
import traceback
sys.path.append('src')

def test_imports():
    """Test if we can import the required modules"""
    print("Testing imports...")
    
    try:
        from face_detection import detect_faces
        print("✅ face_detection imported successfully")
    except Exception as e:
        print(f"❌ face_detection import failed: {e}")
        return False
    
    try:
        from glasses_overlay import overlay_glasses
        print("✅ glasses_overlay imported successfully")
    except Exception as e:
        print(f"❌ glasses_overlay import failed: {e}")
        return False
    
    try:
        from glasses_3d_renderer import overlay_3d_glasses
        print("✅ glasses_3d_renderer imported successfully")
    except Exception as e:
        print(f"❌ glasses_3d_renderer import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_face_detection():
    """Test face detection"""
    print("\nTesting face detection...")
    
    test_image_path = "uploads/carmen-electra-face-profile-earrings-wallpaper-preview.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    try:
        from face_detection import detect_faces
        
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ Could not load test image")
            return False
        
        faces = detect_faces(image, method='dnn', confidence_threshold=0.5)
        print(f"✅ Detected {len(faces)} faces")
        return len(faces) > 0
        
    except Exception as e:
        print(f"❌ Face detection failed: {e}")
        traceback.print_exc()
        return False

def test_2d_glasses():
    """Test 2D glasses overlay"""
    print("\nTesting 2D glasses overlay...")
    
    try:
        from face_detection import detect_faces
        from glasses_overlay import overlay_glasses
        
        test_image_path = "uploads/carmen-electra-face-profile-earrings-wallpaper-preview.jpg"
        image = cv2.imread(test_image_path)
        faces = detect_faces(image, method='dnn', confidence_threshold=0.5)
        
        glasses_path = os.path.join('assets', 'glasses', 'glasses1.png')
        if not os.path.exists(glasses_path):
            print(f"❌ Glasses image not found: {glasses_path}")
            return False
        
        result = overlay_glasses(image, faces, glasses_path)
        cv2.imwrite("test_result_2d.jpg", result)
        print("✅ 2D glasses overlay successful")
        return True
        
    except Exception as e:
        print(f"❌ 2D glasses overlay failed: {e}")
        traceback.print_exc()
        return False

def test_3d_dependencies():
    """Test if 3D dependencies are available"""
    print("\nTesting 3D dependencies...")
    
    try:
        import trimesh
        print("✅ trimesh available")
    except ImportError:
        print("❌ trimesh not available")
        return False
    
    try:
        import pyrender
        print("✅ pyrender available")
    except ImportError:
        print("❌ pyrender not available")
        return False
    
    try:
        import pyglet
        print("✅ pyglet available")
    except ImportError:
        print("❌ pyglet not available")
        return False
    
    return True

def main():
    """Main test function"""
    print("=== Face Glasses App Test Suite ===\n")
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return
    
    # Test face detection
    if not test_face_detection():
        print("❌ Face detection test failed")
        return
    
    # Test 2D glasses
    if not test_2d_glasses():
        print("❌ 2D glasses test failed")
        return
    
    # Test 3D dependencies
    if test_3d_dependencies():
        print("✅ 3D dependencies available")
        # Test 3D glasses
        try:
            from glasses_3d_renderer import overlay_3d_glasses
            from face_detection import detect_faces
            
            test_image_path = "uploads/carmen-electra-face-profile-earrings-wallpaper-preview.jpg"
            image = cv2.imread(test_image_path)
            faces = detect_faces(image, method='dnn', confidence_threshold=0.5)
            
            glb_path = os.path.join('assets', 'glasses', 'glasses.glb')
            if os.path.exists(glb_path):
                print("Testing 3D glasses overlay...")
                result = overlay_3d_glasses(image, faces, glb_path)
                cv2.imwrite("test_result_3d.jpg", result)
                print("✅ 3D glasses overlay successful")
            else:
                print(f"❌ GLB file not found: {glb_path}")
                
        except Exception as e:
            print(f"❌ 3D glasses test failed: {e}")
            traceback.print_exc()
    else:
        print("❌ 3D dependencies not available, skipping 3D test")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
