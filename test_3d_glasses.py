import cv2
import os
import sys
sys.path.append('src')

from face_detection import detect_faces
from glasses_3d_renderer import overlay_3d_glasses

def test_3d_glasses():
    """Test 3D glasses overlay"""
    print("Testing 3D glasses overlay...")
    
    # Check if GLB file exists
    glb_path = os.path.join('assets', 'glasses', 'glasses.glb')
    if not os.path.exists(glb_path):
        print(f"❌ GLB file not found at: {glb_path}")
        print("Please place your glasses.glb file in the assets folder")
        return False
    
    # Test with a sample image (you can replace with your own)
    test_image_path = "uploads/carmen-electra-face-profile-earrings-wallpaper-preview.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please provide a test image with faces")
        return False
    
    # Load and process image
    image = cv2.imread(test_image_path)
    if image is None:
        print("❌ Could not load test image")
        return False
    
    # Detect faces
    faces = detect_faces(image, method='dnn', confidence_threshold=0.5)
    print(f"✅ Detected {len(faces)} faces")
    
    if not faces:
        print("❌ No faces detected in test image")
        return False
    
    # Apply 3D glasses
    result = overlay_3d_glasses(image, faces, glb_path)
    
    # Save result
    cv2.imwrite("test_result_3d.jpg", result)
    print("✅ 3D glasses overlay test completed!")
    print("📁 Result saved as: test_result_3d.jpg")
    
    return True

if __name__ == "__main__":
    test_3d_glasses()