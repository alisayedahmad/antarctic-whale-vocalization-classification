#!/usr/bin/env python3
"""Test installation and verify dependencies."""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'torch': 'torch',
        'hdbscan': 'hdbscan',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'yaml': 'pyyaml',
        'tqdm': 'tqdm',
        'joblib': 'joblib',
        'streamlit': 'streamlit'
    }
    
    failed = []
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - FAILED")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import {len(failed)} packages:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(failed)}")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True


def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = [
        'src',
        'src/data',
        'src/models',
        'src/features',
        'src/evaluation',
        'src/utils',
        'scripts',
        'config',
        'app'
    ]
    
    required_files = [
        'config/config.yaml',
        'requirements.txt',
        'setup.py',
        'README.md'
    ]
    
    project_root = Path(__file__).parent
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
            print(f"✗ Directory missing: {dir_path}")
        else:
            print(f"✓ {dir_path}")
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"✗ File missing: {file_path}")
        else:
            print(f"✓ {file_path}")
    
    if missing_dirs or missing_files:
        print(f"\n❌ Project structure incomplete")
        return False
    else:
        print("\n✅ Project structure is correct!")
        return True


def test_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    
    sys.path.insert(0, str(Path(__file__).parent))
    
    modules = [
        'src.data.preprocessing',
        'src.data.augmentation',
        'src.data.dataset',
        'src.models.svm_classifier',
        'src.models.neural_network',
        'src.models.tree_models',
        'src.models.bayesian_classifier',
        'src.models.clustering',
        'src.features.feature_extraction',
        'src.evaluation.metrics',
        'src.utils.logger',
        'src.utils.visualization'
    ]
    
    failed = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module} - FAILED: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n❌ Failed to import {len(failed)} project modules")
        return False
    else:
        print("\n✅ All project modules can be imported!")
        return True


def main():
    """Run all tests."""
    print("="*60)
    print("Antarctic Whale Vocalization Classification")
    print("Installation Test")
    print("="*60)
    
    results = []
    
    results.append(('Package Imports', test_imports()))
    results.append(('Project Structure', test_project_structure()))
    results.append(('Project Modules', test_modules()))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("1. Download the dataset to data/raw/biodcase_development_set/")
        print("2. Run: python scripts/prepare_data.py")
        print("3. Run: python scripts/train.py --model all")
        print("4. Launch the app: streamlit run app/streamlit_app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
