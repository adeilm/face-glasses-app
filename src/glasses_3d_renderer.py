import cv2
import numpy as np
from PIL import Image
import logging
import os
import json

logger = logging.getLogger(__name__)

class Glasses3DRenderer:
    def __init__(self, glb_path):
        self.glb_path = glb_path
        self.renderer = None
        self.scene = None
        self.initialize_renderer()
    
    def initialize_renderer(self):
        """Initialize the 3D rendering system"""
        try:
            import trimesh
            import pyrender
            
            # Load the 3D glasses model
            self.scene = trimesh.load(self.glb_path)
            logger.info(f"Successfully loaded 3D glasses model: {self.glb_path}")
            
            # Initialize renderer
            self.renderer = pyrender.OffscreenRenderer(400, 400)
            
            return True
            
        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            logger.error("Please install: pip install trimesh pyrender pyglet")
            return False
        except Exception as e:
            logger.error(f"Error initializing 3D renderer: {e}")
            return False
    
    def calculate_face_pose(self, face_landmarks):
        """Calculate face pose from landmarks"""
        try:
            # Extract key points
            left_eye = face_landmarks.get('left_eye', (0, 0))
            right_eye = face_landmarks.get('right_eye', (0, 0))
            nose_tip = face_landmarks.get('nose_tip', (0, 0))
            
            # Calculate face orientation
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            eye_distance = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
            
            # Calculate angles
            roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
            yaw = np.arctan2(nose_tip[0] - eye_center[0], eye_distance) * 0.5
            pitch = np.arctan2(nose_tip[1] - eye_center[1], eye_distance) * 0.3
            
            return {
                'position': eye_center,
                'roll': roll,
                'yaw': yaw,
                'pitch': pitch,
                'scale': eye_distance / 100.0  # Normalize scale
            }
            
        except Exception as e:
            logger.error(f"Error calculating face pose: {e}")
            return None
    
    def render_glasses(self, face_pose, image_size):
        """Render 3D glasses for given face pose"""
        try:
            if not self.renderer or not self.scene:
                logger.error("Renderer not initialized")
                return None
            
            import pyrender
            import trimesh
            
            # Create rendering scene
            render_scene = pyrender.Scene()
            
            # Convert trimesh to pyrender mesh
            if isinstance(self.scene, trimesh.Trimesh):
                mesh = pyrender.Mesh.from_trimesh(self.scene)
            else:
                # Handle scene with multiple meshes
                geometries = list(self.scene.geometry.values())
                if geometries:
                    mesh = pyrender.Mesh.from_trimesh(geometries[0])
                else:
                    logger.error("No geometry found in GLB file")
                    return None
            
            # Create transformation matrix
            transform = np.eye(4)
            
            # Apply scaling
            scale = face_pose['scale']
            transform[:3, :3] *= scale
            
            # Apply rotation
            roll, yaw, pitch = face_pose['roll'], face_pose['yaw'], face_pose['pitch']
            
            # Rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ])
            
            Ry = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ])
            
            Rz = np.array([
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1]
            ])
            
            # Apply rotations
            rotation = Rz @ Ry @ Rx
            transform[:3, :3] = transform[:3, :3] @ rotation
            
            # Add mesh to scene
            render_scene.add(mesh, pose=transform)
            
            # Set up camera
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            camera_pose = np.eye(4)
            camera_pose[2, 3] = 1.5  # Move camera back
            render_scene.add(camera, pose=camera_pose)
            
            # Add lighting
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            render_scene.add(light, pose=camera_pose)
            
            # Render
            color, depth = self.renderer.render(render_scene)
            
            # Convert to PIL Image with alpha channel
            alpha = (depth > 0).astype(np.uint8) * 255
            
            # Resize to match face size
            glasses_image = Image.fromarray(color)
            glasses_image.putalpha(Image.fromarray(alpha))
            
            return glasses_image
            
        except Exception as e:
            logger.error(f"Error rendering 3D glasses: {e}")
            return None
    
    def cleanup(self):
        """Clean up renderer resources"""
        if self.renderer:
            self.renderer.delete()

def detect_face_landmarks(image, face_rect):
    """Detect facial landmarks for 3D positioning"""
    try:
        import dlib
        
        # Initialize dlib predictor
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            logger.warning("dlib landmarks file not found, using estimated positions")
            return estimate_landmarks_from_face(face_rect)
        
        predictor = dlib.shape_predictor(predictor_path)
        
        # Convert face rectangle to dlib format
        x, y, w, h = face_rect[:4]
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get landmarks
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray, rect)
        
        # Extract key points
        landmarks_dict = {
            'left_eye': ((landmarks.part(36).x + landmarks.part(39).x) // 2,
                        (landmarks.part(36).y + landmarks.part(39).y) // 2),
            'right_eye': ((landmarks.part(42).x + landmarks.part(45).x) // 2,
                         (landmarks.part(42).y + landmarks.part(45).y) // 2),
            'nose_tip': (landmarks.part(30).x, landmarks.part(30).y),
            'nose_bridge': (landmarks.part(27).x, landmarks.part(27).y),
        }
        
        return landmarks_dict
        
    except ImportError:
        logger.warning("dlib not available, using estimated landmarks")
        return estimate_landmarks_from_face(face_rect)
    except Exception as e:
        logger.error(f"Error detecting landmarks: {e}")
        return estimate_landmarks_from_face(face_rect)

def estimate_landmarks_from_face(face_rect):
    """Estimate facial landmarks from face rectangle"""
    x, y, w, h = face_rect[:4]
    
    return {
        'left_eye': (x + int(w * 0.3), y + int(h * 0.35)),
        'right_eye': (x + int(w * 0.7), y + int(h * 0.35)),
        'nose_tip': (x + int(w * 0.5), y + int(h * 0.55)),
        'nose_bridge': (x + int(w * 0.5), y + int(h * 0.4)),
    }

def overlay_3d_glasses(image, face_coordinates, glb_path):
    """
    Main function to overlay 3D glasses on detected faces
    
    Args:
        image: OpenCV image (BGR format)
        face_coordinates: List of face detection results
        glb_path: Path to the .glb glasses file
    
    Returns:
        OpenCV image with 3D glasses overlaid
    """
    try:
        # Initialize 3D renderer
        renderer = Glasses3DRenderer(glb_path)
        
        if not renderer.renderer:
            logger.error("Failed to initialize 3D renderer")
            # Fallback to 2D glasses
            from glasses_overlay import overlay_glasses
            return overlay_glasses(image, face_coordinates, "assets/glasses.png")
        
        # Convert to PIL for easier manipulation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        for face_data in face_coordinates:
            # Get face landmarks
            landmarks = detect_face_landmarks(image, face_data)
            
            # Calculate face pose
            face_pose = renderer.calculate_face_pose(landmarks)
            
            if face_pose:
                # Render 3D glasses
                glasses_rendered = renderer.render_glasses(face_pose, (200, 200))
                
                if glasses_rendered:
                    # Position glasses on face
                    x, y = face_pose['position']
                    glasses_x = int(x - glasses_rendered.width // 2)
                    glasses_y = int(y - glasses_rendered.height // 2)
                    
                    # Ensure bounds
                    glasses_x = max(0, min(glasses_x, pil_image.width - glasses_rendered.width))
                    glasses_y = max(0, min(glasses_y, pil_image.height - glasses_rendered.height))
                    
                    # Overlay glasses
                    if glasses_rendered.mode == 'RGBA':
                        pil_image.paste(glasses_rendered, (glasses_x, glasses_y), glasses_rendered)
                    else:
                        pil_image.paste(glasses_rendered, (glasses_x, glasses_y))
        
        # Clean up
        renderer.cleanup()
        
        # Convert back to OpenCV format
        result_array = np.array(pil_image)
        result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
        
        logger.info("Successfully applied 3D glasses overlay")
        return result_bgr
        
    except Exception as e:
        logger.error(f"Error in 3D glasses overlay: {e}")
        # Fallback to 2D glasses
        from glasses_overlay import overlay_glasses
        return overlay_glasses(image, face_coordinates, "assets/glasses.png")