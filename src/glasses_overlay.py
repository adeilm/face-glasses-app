import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def overlay_glasses(image, face_coordinates, glasses_path):
    """
    Overlay glasses on detected faces with improved positioning
    Args:
        image: OpenCV image (BGR format)
        face_coordinates: List of (x, y, w, h) or (x, y, w, h, confidence) tuples for detected faces
        glasses_path: Path to the glasses image file
    Returns:
        OpenCV image with glasses overlaid
    """
    try:
        # Convert OpenCV image to PIL for easier manipulation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Load glasses image
        glasses = Image.open(glasses_path)
        
        for face_data in face_coordinates:
            # Handle both formats: (x, y, w, h) and (x, y, w, h, confidence)
            if len(face_data) == 5:
                x, y, w, h, confidence = face_data
                # Only process high-confidence faces
                if confidence < 0.3:
                    continue
                logger.info(f"Processing face with confidence: {confidence:.2f}")
            else:
                x, y, w, h = face_data
                confidence = 1.0
            
            # More precise glasses sizing - adjust based on confidence
            confidence_factor = min(confidence, 1.0) if confidence < 1.0 else 1.0
            glasses_width = int(w * (0.7 + 0.1 * confidence_factor))  # 70-80% of face width
            glasses_height = int(glasses_width * glasses.height / glasses.width)
            
            # Resize glasses
            glasses_resized = glasses.resize((glasses_width, glasses_height), Image.Resampling.LANCZOS)
            
            # More accurate positioning based on facial proportions
            glasses_x = x + int(w * 0.1)  # Center horizontally with 10% margin
            glasses_y = y + int(h * 0.25)  # Position at eye level (25% from top)
            
            # Fine-tune vertical position based on glasses height
            glasses_y = max(glasses_y - int(glasses_height * 0.1), y)
            
            # Ensure glasses don't go outside image bounds
            if glasses_x + glasses_width > pil_image.width:
                glasses_width = pil_image.width - glasses_x
                glasses_height = int(glasses_width * glasses.height / glasses.width)
                glasses_resized = glasses_resized.resize((glasses_width, glasses_height), Image.Resampling.LANCZOS)
            
            if glasses_y + glasses_height > pil_image.height:
                glasses_height = pil_image.height - glasses_y
                glasses_width = int(glasses_height * glasses.width / glasses.height)
                glasses_resized = glasses_resized.resize((glasses_width, glasses_height), Image.Resampling.LANCZOS)
            
            # Ensure minimum bounds
            glasses_x = max(0, glasses_x)
            glasses_y = max(0, glasses_y)
            
            # Overlay glasses with transparency support
            if glasses_resized.mode == 'RGBA':
                pil_image.paste(glasses_resized, (glasses_x, glasses_y), glasses_resized)
            else:
                pil_image.paste(glasses_resized, (glasses_x, glasses_y))
        
        # Convert back to OpenCV format
        result_array = np.array(pil_image)
        result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        
        logger.info("Successfully overlaid glasses on faces")
        return result_bgr
        
    except Exception as e:
        logger.error(f"Error in glasses overlay: {e}")
        return image

def overlay_glasses_with_landmarks(image, face_coordinates, glasses_path):
    """
    Enhanced version using eye detection for more accurate positioning
    Args:
        image: OpenCV image (BGR format)
        face_coordinates: List of (x, y, w, h) tuples for detected faces
        glasses_path: Path to the glasses image file
        face_cascade: Optional eye cascade classifier for better positioning
    Returns:
        OpenCV image with glasses overlaid
    """
    try:
        # Convert OpenCV image to PIL for easier manipulation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Load glasses image
        glasses = Image.open(glasses_path)
        
        # Load eye cascade
        try:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except:
            eye_cascade = None
            logger.warning("Eye cascade not available, using basic positioning")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for face_data in face_coordinates:
            # Handle both formats
            if len(face_data) == 5:
                x, y, w, h, confidence = face_data
                if confidence < 0.3:
                    continue
            else:
                x, y, w, h = face_data
                confidence = 1.0
            
            # Try to detect eyes within the face region
            roi_gray = gray[y:y+h, x:x+w]
            eyes = []
            
            if eye_cascade is not None:
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(eyes) >= 2:
                # Use eye positions for more accurate placement
                eye_centers = []
                for (ex, ey, ew, eh) in eyes:
                    eye_center_x = x + ex + ew // 2
                    eye_center_y = y + ey + eh // 2
                    eye_centers.append((eye_center_x, eye_center_y))
                
                # Sort eyes by x-coordinate and take the two most likely candidates
                eye_centers.sort(key=lambda point: point[0])
                
                if len(eye_centers) >= 2:
                    left_eye = eye_centers[0]
                    right_eye = eye_centers[1]
                    
                    # Calculate glasses dimensions based on eye distance
                    eye_distance = abs(right_eye[0] - left_eye[0])
                    glasses_width = int(eye_distance * 2.5)  # Glasses wider than eye distance
                    glasses_height = int(glasses_width * glasses.height / glasses.width)
                    
                    # Position glasses centered on eyes
                    glasses_x = left_eye[0] - int(glasses_width * 0.2)
                    glasses_y = min(left_eye[1], right_eye[1]) - int(glasses_height * 0.4)
                else:
                    # Fallback to original method
                    glasses_width = int(w * 0.8)
                    glasses_height = int(glasses_width * glasses.height / glasses.width)
                    glasses_x = x + int(w * 0.1)
                    glasses_y = y + int(h * 0.25)
            else:
                # Fallback to original method
                glasses_width = int(w * 0.8)
                glasses_height = int(glasses_width * glasses.height / glasses.width)
                glasses_x = x + int(w * 0.1)
                glasses_y = y + int(h * 0.25)
            
            # Resize glasses
            glasses_resized = glasses.resize((glasses_width, glasses_height), Image.Resampling.LANCZOS)
            
            # Ensure glasses don't go outside image bounds
            glasses_x = max(0, min(glasses_x, pil_image.width - glasses_width))
            glasses_y = max(0, min(glasses_y, pil_image.height - glasses_height))
            
            # Overlay glasses with transparency support
            if glasses_resized.mode == 'RGBA':
                pil_image.paste(glasses_resized, (glasses_x, glasses_y), glasses_resized)
            else:
                pil_image.paste(glasses_resized, (glasses_x, glasses_y))
        
        # Convert back to OpenCV format
        result_array = np.array(pil_image)
        result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        
        return result_bgr
        
    except Exception as e:
        logger.error(f"Error in enhanced glasses overlay: {e}")
        return overlay_glasses(image, face_coordinates, glasses_path)