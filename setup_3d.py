import subprocess
import sys
import os

def install_3d_dependencies():
    """Install required 3D libraries"""
    try:
        print("Installing 3D rendering dependencies...")
        
        # Install core 3D libraries
        subprocess.check_call([sys.executable, "-m", "pip", "install", "trimesh==4.4.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyrender==0.1.45"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyglet==2.0.10"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pygltflib==1.16.1"])
        
        # Try to install dlib (optional)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "dlib==19.24.2"])
            print("✅ dlib installed successfully")
        except:
            print("⚠️  dlib installation failed - using estimated landmarks")
        
        print("✅ 3D dependencies installed successfully!")
        
        # Create assets directory if it doesn't exist
        assets_dir = os.path.join("assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        print("\n📁 Please place your glasses.glb file in the 'assets' folder")
        print("📁 Also ensure you have glasses.png as a fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ Error installing 3D dependencies: {e}")
        return False

if __name__ == "__main__":
    install_3d_dependencies()