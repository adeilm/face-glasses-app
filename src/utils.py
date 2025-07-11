import cv2
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

def validate_image_file(file_path):
    """Validate if file is a valid image"""
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False
        
        # Check file size (max 16MB)
        if os.path.getsize(file_path) > 16 * 1024 * 1024:
            return False
        
        # Try to open with PIL
        with Image.open(file_path) as img:
            img.verify()
        
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

def load_image(file_path):
    """Load image using OpenCV for face detection compatibility"""
    try:
        if not validate_image_file(file_path):
            logger.error(f"Invalid image file: {file_path}")
            return None
        
        image = cv2.imread(file_path)
        if image is None:
            logger.error(f"Failed to load image: {file_path}")
            return None
        
        logger.info(f"Successfully loaded image: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None

def save_image(image, output_path):
    """Save image - handles both OpenCV and PIL formats"""
    try:
        if isinstance(image, np.ndarray):
            # OpenCV format
            success = cv2.imwrite(output_path, image)
            if not success:
                raise Exception("Failed to save image with OpenCV")
        else:
            # PIL format
            image.save(output_path)
        
        logger.info(f"Image saved successfully: {output_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise