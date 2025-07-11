import cv2
import numpy as np
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# Model URLs for download
MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

# Local model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_FILE = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
CONFIG_FILE = os.path.join(MODEL_DIR, "deploy.prototxt")

def download_models():
    """Download DNN models if they don't exist"""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Download model file if not exists
        if not os.path.exists(MODEL_FILE):
            logger.info("Downloading face detection model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
            logger.info("Model downloaded successfully")
        
        # Download config file if not exists
        if not os.path.exists(CONFIG_FILE):
            logger.info("Downloading model configuration...")
            urllib.request.urlretrieve(CONFIG_URL, CONFIG_FILE)
            logger.info("Configuration downloaded successfully")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        return False

def detect_faces_dnn(image, confidence_threshold=0.5):
    """
    Detect faces using OpenCV DNN module with a pretrained SSD model.

    Args:
        image: Input image (BGR format).
        confidence_threshold: Minimum confidence to filter weak detections.

    Returns:
        List of (x, y, w, h, confidence) for each detected face.
    """
    try:
        if image is None:
            logger.error("Invalid image input")
            return []

        # Download models if needed
        if not download_models():
            logger.warning("Failed to download DNN models, falling back to Haar cascades")
            return detect_faces_haar(image)

        h, w = image.shape[:2]

        # Load pre-trained model
        net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

        # Prepare the input blob
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)

        detections = net.forward()
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x, y = max(0, x1), max(0, y1)
                w_box, h_box = x2 - x1, y2 - y1
                
                # Ensure positive dimensions
                if w_box > 0 and h_box > 0:
                    faces.append((x, y, w_box, h_box, float(confidence)))

        logger.info(f"DNN detected {len(faces)} faces")
        return faces

    except Exception as e:
        logger.error(f"DNN face detection error: {e}, falling back to Haar cascades")
        return detect_faces_haar(image)

def detect_faces_haar(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Fallback face detection using Haar cascades (your original method)
    
    Args:
        image: OpenCV image (BGR format)
        scale_factor: Parameter specifying how much the image size is reduced at each scale
        min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it
        min_size: Minimum possible object size, smaller objects are ignored
        
    Returns:
        List of (x, y, w, h) tuples for detected faces (without confidence)
    """
    try:
        if image is None:
            logger.error("Invalid image provided to face detection")
            return []
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            return []
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        
        # Convert to format with confidence (set to 1.0 for Haar cascades)
        faces_with_confidence = []
        for (x, y, w, h) in faces:
            faces_with_confidence.append((x, y, w, h, 1.0))
        
        logger.info(f"Haar detected {len(faces)} faces")
        return faces_with_confidence
        
    except Exception as e:
        logger.error(f"Error in Haar face detection: {e}")
        return []

def detect_faces(image, method='dnn', confidence_threshold=0.5):
    """
    Main face detection function with multiple methods
    
    Args:
        image: OpenCV image (BGR format)
        method: 'dnn' for deep learning or 'haar' for Haar cascades
        confidence_threshold: Minimum confidence for DNN detection
        
    Returns:
        List of (x, y, w, h, confidence) tuples for detected faces
    """
    if method == 'dnn':
        return detect_faces_dnn(image, confidence_threshold)
    else:
        return detect_faces_haar(image)

def detect_faces_with_confidence(image, confidence_threshold=0.7):
    """
    Enhanced face detection with higher confidence threshold
    """
    return detect_faces_dnn(image, confidence_threshold)

def draw_faces(image, faces):
    """
    Draw bounding boxes around detected faces
    
    Args:
        image: Input image
        faces: List of (x, y, w, h, confidence) tuples
        
    Returns:
        Image with drawn bounding boxes
    """
    image_copy = image.copy()
    
    for face_data in faces:
        if len(face_data) == 5:  # With confidence
            x, y, w, h, conf = face_data
            label = f"{conf:.2f}"
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)  # Green for high confidence, orange for low
        else:  # Without confidence (Haar cascades)
            x, y, w, h = face_data
            label = "Face"
            color = (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        cv2.putText(image_copy, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image_copy

def get_best_faces(faces, max_faces=5):
    """
    Get the best faces based on confidence and size
    
    Args:
        faces: List of detected faces with confidence
        max_faces: Maximum number of faces to return
        
    Returns:
        List of best faces sorted by confidence
    """
    if not faces:
        return []
    
    # Sort by confidence (descending) and face area (descending)
    faces_sorted = sorted(faces, key=lambda f: (f[4] if len(f) == 5 else 1.0, f[2] * f[3]), reverse=True)
    
    return faces_sorted[:max_faces]