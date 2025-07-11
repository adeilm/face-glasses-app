# Face Glasses App - Clean Architecture

## 🏗️ Project Structure

```
face-glasses-app/
├── 📁 src/                          # Core application code
│   ├── app.py                       # Flask web application
│   ├── face_detection.py            # Face detection (OpenCV DNN + Haar)
│   ├── glasses_overlay.py           # 2D glasses overlay
│   ├── glasses_3d_renderer.py       # 3D glasses overlay (legacy)
│   ├── utils.py                     # Utility functions
│   └── __init__.py                  # Package initialization
│
├── 📁 core/                         # Advanced 3D system (NEW)
│   ├── mediapipe_3d_glasses.py      # MediaPipe-based 3D glasses
│   ├── simple_3d_glasses.py         # Simplified 3D glasses
│   └── __init__.py                  # Package initialization
│
├── 📁 assets/                       # Static assets
│   └── glasses/
│       ├── glasses.glb              # 3D glasses model
│       └── glasses1.png             # 2D glasses image
│
├── 📁 models/                       # AI/ML models
│   ├── deploy.prototxt              # OpenCV DNN config
│   └── res10_300x300_ssd_iter_140000.caffemodel  # DNN model
│
├── 📁 templates/                    # HTML templates
│   └── index.html                   # Web interface
│
├── 📁 uploads/                      # User uploaded images
│   └── (user images)
│
├── 📁 output/                       # Generated results
│   └── (result images)
│
├── 📁 tests/                        # Test files (NEW)
│   ├── test_core_systems.py         # Main test suite
│   ├── test_mediapipe.py            # MediaPipe system tests
│   └── debug_tools.py               # Debug utilities
│
├── 📁 docs/                         # Documentation (NEW)
│   ├── API.md                       # API documentation
│   ├── SETUP.md                     # Setup instructions
│   └── ARCHITECTURE.md              # Architecture overview
│
├── requirements.txt                 # Python dependencies
├── requirements_mediapipe.txt       # MediaPipe-specific deps
├── main.py                          # Main entry point
└── README.md                        # Project overview
```

## 🚀 Essential Files

### Core Application (src/)
- **app.py**: Flask web application with both 2D and 3D overlay support
- **face_detection.py**: Robust face detection using OpenCV DNN and Haar cascades
- **glasses_overlay.py**: 2D glasses overlay system
- **glasses_3d_renderer.py**: 3D glasses overlay using trimesh/pyrender
- **utils.py**: Image processing utilities

### Advanced 3D System (core/)
- **mediapipe_3d_glasses.py**: Advanced MediaPipe-based 3D glasses with 468 landmarks
- **simple_3d_glasses.py**: Simplified 3D glasses with better visibility

### Assets & Models
- **assets/glasses/glasses.glb**: 3D glasses model
- **assets/glasses/glasses1.png**: 2D glasses fallback
- **models/**: OpenCV DNN models for face detection

### Testing & Debug
- **tests/**: Comprehensive test suite
- **main.py**: Single entry point for all functionality

## 🧹 Files to Remove

### Redundant/Temporary Files
- All test_*.py files in root (will be moved to tests/)
- debug_*.py files in root (will be moved to tests/)
- create_glasses.py (utility, can be recreated)
- setup_3d.py (replaced by requirements)
- __pycache__/ directories
- All .jpg result files in root (will be moved to output/)

## 🎯 Usage

1. **Web Application**: `python main.py --mode web`
2. **Command Line**: `python main.py --mode cli --input image.jpg`
3. **Real-time Webcam**: `python main.py --mode webcam`
4. **Run Tests**: `python main.py --mode test`

## 🔧 System Features

### 1. Face Detection
- OpenCV DNN (high accuracy)
- Haar Cascades (fast fallback)
- Confidence thresholding

### 2. 2D Glasses Overlay
- Simple PNG overlay
- Automatic scaling and positioning
- Real-time processing

### 3. 3D Glasses Overlay
- **Legacy**: trimesh + pyrender rendering
- **Advanced**: MediaPipe Face Mesh (468 landmarks)
- **Simple**: OpenCV-based 3D effect

### 4. Web Interface
- Drag & drop image upload
- Real-time preview
- Multiple rendering modes

## 🎨 Architecture Benefits

1. **Modular**: Clear separation of concerns
2. **Scalable**: Easy to add new features
3. **Testable**: Comprehensive test coverage
4. **Maintainable**: Well-organized codebase
5. **Flexible**: Multiple rendering options
