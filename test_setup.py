"""Test script to verify setup and dependencies"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        ('google.genai', 'google-genai'),
        ('pycolmap', 'pycolmap'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('qwen_vl_utils', 'qwen-vl-utils'),
        ('gradio_client', 'gradio-client'),
        ('cv2', 'opencv-python'),
        ('PIL', 'pillow'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('open3d', 'open3d'),
        ('timm', 'timm'),
        ('yaml', 'pyyaml'),
        ('dotenv', 'python-dotenv'),
        ('tqdm', 'tqdm'),
    ]
    
    optional_packages = [
        ('soundfile', 'soundfile'),
        ('librosa', 'librosa'),
    ]
    
    missing_required = []
    missing_optional = []
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (REQUIRED)")
            missing_required.append(package_name)
    
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name} (optional)")
        except ImportError:
            print(f"  ⚠ {package_name} (optional)")
            missing_optional.append(package_name)
    
    return missing_required, missing_optional


def test_directories():
    """Test that all required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = [
        'data/input',
        'data/generated',
        'data/reconstruction',
        'data/segmentation',
        'data/audio',
        'data/output',
        'src',
        'notebooks'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path}")
            missing_dirs.append(dir_path)
    
    return missing_dirs


def test_files():
    """Test that all required files exist"""
    print("\nTesting files...")
    
    required_files = [
        'main.py',
        'config.yaml',
        'requirements.txt',
        'README.md',
        '.env.example',
        'src/__init__.py',
        'src/utils.py',
        'src/image_generation.py',
        'src/reconstruction.py',
        'src/segmentation.py',
        'src/audio_generation.py',
        'src/spatial_audio.py',
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}")
            missing_files.append(file_path)
    
    return missing_files


def test_env_variables():
    """Test environment variables"""
    print("\nTesting environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['GEMINI_API_KEY']
    optional_vars = ['HF_TOKEN', 'DEVICE', 'MODEL_CACHE_DIR']
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var} = {value[:10]}..." if len(value) > 10 else f"  ✓ {var} = {value}")
        else:
            print(f"  ✗ {var} (REQUIRED)")
            missing_vars.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var} = {value}")
        else:
            print(f"  ⚠ {var} (optional, not set)")
    
    return missing_vars


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {str(e)}")


def main():
    """Run all tests"""
    print("=" * 80)
    print("Multi-Modal Scene Audio Pipeline - Setup Test")
    print("=" * 80)
    
    # Test imports
    missing_required, missing_optional = test_imports()
    
    # Test directories
    missing_dirs = test_directories()
    
    # Test files
    missing_files = test_files()
    
    # Test environment variables
    missing_vars = test_env_variables()
    
    # Test CUDA
    test_cuda()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_good = True
    
    if missing_required:
        print(f"✗ Missing required packages: {', '.join(missing_required)}")
        print(f"  Install with: pip install {' '.join(missing_required)}")
        all_good = False
    else:
        print("✓ All required packages installed")
    
    if missing_optional:
        print(f"⚠ Missing optional packages: {', '.join(missing_optional)}")
        print(f"  Install with: pip install {' '.join(missing_optional)}")
    
    if missing_dirs:
        print(f"✗ Missing directories: {', '.join(missing_dirs)}")
        all_good = False
    else:
        print("✓ All required directories exist")
    
    if missing_files:
        print(f"✗ Missing files: {', '.join(missing_files)}")
        all_good = False
    else:
        print("✓ All required files exist")
    
    if missing_vars:
        print(f"✗ Missing environment variables: {', '.join(missing_vars)}")
        print("  Copy .env.example to .env and fill in your API keys")
        all_good = False
    else:
        print("✓ All required environment variables set")
    
    print("=" * 80)
    
    if all_good:
        print("\n✓✓✓ Setup complete! You're ready to run the pipeline. ✓✓✓")
        print("\nNext steps:")
        print("  1. Place your input images in data/input/")
        print("  2. Run: python main.py")
        print("  3. Or use the Jupyter notebook: jupyter notebook notebooks/pipeline_demo.ipynb")
        return 0
    else:
        print("\n✗✗✗ Setup incomplete. Please fix the issues above. ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
