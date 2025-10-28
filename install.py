#!/usr/bin/env python3
"""Installation automatique avec gestion d'erreurs."""

import subprocess
import sys
import os

def run_command(cmd):
    """Execute une commande et retourne le succès."""
    try:
        subprocess.run(cmd, check=True, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("="*60)
    print("Installation Antarctic Whale Classification")
    print("="*60)
    
    # Upgrade pip
    print("\n[1/4] Upgrade pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Installation minimale d'abord
    print("\n[2/4] Installation des packages essentiels...")
    essential = [
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scipy==1.11.4",
        "pyyaml==6.0.1",
        "tqdm==4.66.1",
        "joblib==1.3.2"
    ]
    
    for pkg in essential:
        print(f"  Installing {pkg}...")
        if not run_command(f"{sys.executable} -m pip install {pkg}"):
            print(f"  ⚠️  Warning: {pkg} failed, continuing...")
    
    # Installation ML packages
    print("\n[3/4] Installation ML packages...")
    ml_packages = [
        "scikit-learn==1.3.2",
        "xgboost==2.0.3",
        "lightgbm==4.1.0"
    ]
    
    for pkg in ml_packages:
        print(f"  Installing {pkg}...")
        if not run_command(f"{sys.executable} -m pip install {pkg}"):
            print(f"  ⚠️  Warning: {pkg} failed, continuing...")
    
    # Installation autres packages
    print("\n[4/4] Installation autres packages...")
    other_packages = [
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "opencv-python-headless==4.8.1.78",
        "Pillow==10.1.0",
        "streamlit==1.29.0",
        "librosa==0.10.1",
        "soundfile==0.12.1",
        "hdbscan==0.8.33"
    ]
    
    failed = []
    for pkg in other_packages:
        print(f"  Installing {pkg}...")
        if not run_command(f"{sys.executable} -m pip install {pkg}"):
            failed.append(pkg)
            print(f"  ⚠️  Warning: {pkg} failed")
    
    print("\n" + "="*60)
    print("Installation terminée!")
    print("="*60)
    
    if failed:
        print("\n⚠️  Packages non installés:")
        for pkg in failed:
            print(f"  - {pkg}")
        print("\nVous pouvez les installer manuellement:")
        print(f"pip install {' '.join(failed)}")
    else:
        print("\n✅ Tous les packages sont installés!")
    
    print("\n📝 Note: PyTorch est optionnel (seulement pour Neural Network)")
    print("Pour l'installer: pip install torch")
    
    print("\n🧪 Test de l'installation:")
    print("python test_install.py")

if __name__ == "__main__":
    main()
