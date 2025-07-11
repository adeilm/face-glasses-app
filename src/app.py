from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import sys
import logging
import time
import cv2
from werkzeug.utils import secure_filename

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detection import detect_faces
from glasses_overlay import overlay_glasses
from utils import load_image, save_image, validate_image_file
from glasses_3d_renderer import overlay_3d_glasses

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Get the parent directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(project_root, 'output')
app.config['ASSETS_FOLDER'] = os.path.join(project_root, 'assets')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Validate request
        if 'image' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not file.filename.lower().endswith(tuple(f'.{ext}' for ext in ALLOWED_EXTENSIONS)):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
        
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Process the image
        image = load_image(file_path)
        if image is None:
            logger.error("Failed to load image")
            return jsonify({'error': 'Failed to load image'}), 400
        
        face_coordinates = detect_faces(image)
        logger.info(f"Detected {len(face_coordinates)} faces")
        
        if len(face_coordinates) == 0:
            return jsonify({'error': 'No faces detected in the image'}), 400
        
        # Check if 3D glasses are available
        glb_path = os.path.join(app.config['ASSETS_FOLDER'], 'glasses', 'glasses.glb')
        
        if os.path.exists(glb_path):
            logger.info("Applying 3D glasses overlay")
            try:
                output_image = overlay_3d_glasses(image, face_coordinates, glb_path)
            except Exception as e:
                logger.error(f"3D overlay failed: {e}, falling back to 2D")
                # Fallback to 2D
                glasses_2d_path = os.path.join(app.config['ASSETS_FOLDER'], 'glasses', 'glasses1.png')
                output_image = overlay_glasses(image, face_coordinates, glasses_2d_path)
        else:
            # Use 2D glasses overlay
            glasses_2d_path = os.path.join(app.config['ASSETS_FOLDER'], 'glasses', 'glasses1.png')
            
            if not os.path.exists(glasses_2d_path):
                logger.error(f"Glasses image not found: {glasses_2d_path}")
                return jsonify({'error': 'Glasses image not found'}), 500
            
            logger.info("Applying 2D glasses overlay")
            output_image = overlay_glasses(image, face_coordinates, glasses_2d_path)
        
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_' + file.filename)
        save_image(output_image, output_path)
        logger.info(f"Output saved: {output_path}")
        
        return send_from_directory(app.config['OUTPUT_FOLDER'], 'output_' + file.filename)
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not validate_image_file(file):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Read image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        # Detect faces using DNN
        faces = detect_faces(image, method='dnn', confidence_threshold=0.5)
        
        if not faces:
            logger.warning("No faces detected in image")
            return jsonify({'error': 'No faces detected in the image'}), 400
        
        # Apply 3D glasses overlay
        glb_path = os.path.join(app.config['ASSETS_FOLDER'], 'glasses', 'glasses.glb')
        
        if os.path.exists(glb_path):
            logger.info("Applying 3D glasses overlay")
            try:
                result_image = overlay_3d_glasses(image, faces, glb_path)
            except Exception as e:
                logger.error(f"3D overlay failed: {e}, falling back to 2D")
                # Fallback to 2D
                glasses_2d_path = os.path.join(app.config['ASSETS_FOLDER'], 'glasses', 'glasses1.png')
                result_image = overlay_glasses(image, faces, glasses_2d_path)
        else:
            logger.warning("GLB file not found, using 2D glasses")
            glasses_path = os.path.join(app.config['ASSETS_FOLDER'], 'glasses', 'glasses1.png')
            result_image = overlay_glasses(image, faces, glasses_path)
        
        # Save result
        result_filename = f"result_{unique_filename}"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_image)
        
        logger.info(f"Successfully processed image with {len(faces)} faces using 3D glasses")
        
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'result_image': f"/uploads/{result_filename}",
            'method': '3D' if os.path.exists(glb_path) else '2D'
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_best_faces(faces, max_faces=10):
    """Get the best faces based on confidence"""
    if not faces:
        return []
    
    # Sort faces by confidence if available
    if len(faces) > 0 and len(faces[0]) > 4:
        faces_sorted = sorted(faces, key=lambda x: x[4] if len(x) > 4 else 0, reverse=True)
    else:
        faces_sorted = faces
    
    return faces_sorted[:max_faces]

if __name__ == '__main__':
    app.run(debug=True)