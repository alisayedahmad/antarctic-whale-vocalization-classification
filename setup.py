from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="antarctic-whale-classification",
    version="1.0.0",
    author="Machine Learning Course Project",
    description="Multi-class classification system for Antarctic whale vocalizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/antarctic-whale-vocalization-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "soundfile>=0.11.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "hdbscan>=0.8.27",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "streamlit>=1.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ]
    },
)
