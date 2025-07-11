#!/usr/bin/env python3
"""
Quick test of the simple 3D glasses system
"""

import cv2
from simple_3d_glasses import SimpleMediaPipe3DGlasses

def test_simple_glasses():
    """Test the simple glasses system"""
    print("Testing Simple 3D Glasses System...")
    
    # Initialize system
    glasses_system = SimpleMediaPipe3DGlasses(show_debug=True)
    
    # Test with existing image
    test_image = "uploads/carmen-electra-face-profile-earrings-wallpaper-preview.jpg"
    output_path = "simple_glasses_test_result.jpg"
    
    print(f"Processing image: {test_image}")
    result = glasses_system.process_image(test_image, output_path)
    
    if result is not None:
        print(f"✅ Success! Result saved to: {output_path}")
        
        # Display result
        cv2.imshow("Simple 3D Glasses Result", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("❌ Failed to process image")
    
    # Cleanup
    glasses_system.cleanup()

if __name__ == "__main__":
    test_simple_glasses()
