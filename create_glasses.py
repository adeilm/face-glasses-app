#!/usr/bin/env python3

import cv2
import numpy as np
import os

def create_simple_glasses():
    """Create a simple 2D glasses image for testing"""
    print("Creating simple 2D glasses image...")
    
    # Create a transparent image
    img = np.zeros((100, 300, 4), dtype=np.uint8)
    
    # Draw glasses frames
    # Left lens
    cv2.circle(img, (75, 50), 35, (0, 0, 0, 255), 3)
    # Right lens
    cv2.circle(img, (225, 50), 35, (0, 0, 0, 255), 3)
    # Bridge
    cv2.line(img, (110, 50), (190, 50), (0, 0, 0, 255), 3)
    # Left temple
    cv2.line(img, (40, 50), (20, 45), (0, 0, 0, 255), 3)
    # Right temple
    cv2.line(img, (260, 50), (280, 45), (0, 0, 0, 255), 3)
    
    # Save as PNG
    glasses_path = os.path.join('assets', 'glasses', 'glasses1.png')
    cv2.imwrite(glasses_path, img)
    print(f"✅ Created simple glasses image: {glasses_path}")
    
    return glasses_path

if __name__ == "__main__":
    create_simple_glasses()
