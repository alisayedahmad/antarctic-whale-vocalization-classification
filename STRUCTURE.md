# Project Structure

This document describes the complete structure of the Antarctic Whale Vocalization Classification project.

## Directory Tree

```
antarctic-whale-vocalization-classification/
│
├── README.md                    # Main project documentation
├── QUICKSTART.md               # Quick start guide
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation script
├── test_install.py            # Installation test script
│
├── config/                     # Configuration files
│   └── config.yaml            # Main configuration file
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── data/                   # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Audio and image preprocessing
│   │   ├── augmentation.py    # Data augmentation techniques
│   │   └── dataset.py         # PyTorch dataset classes
│   │
│   ├── models/                 # Machine learning models
│   │   ├── __init__.py
│   │   ├── svm_classifier.py       # SVM implementation
│   │   ├── neural_network.py       # Neural network implementation
│   │   ├── bayesian_classifier.py  # Naive Bayes implementation
│   │   ├── tree_models.py          # Tree-based models (RF, XGBoost, LightGBM)
│   │   └── clustering.py           # Clustering methods
│   │
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_extraction.py  # Feature extraction utilities
│   │
│   ├── evaluation/             # Model evaluation
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics and visualization
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── logger.py          # Logging utilities
│       └── visualization.py   # Visualization utilities
│
├── scripts/                    # Executable scripts
│   ├── prepare_data.py        # Data preparation pipeline
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Model evaluation script
│   └── inference.py           # Inference script
│
├── app/                        # Web application
│   └── streamlit_app.py       # Streamlit interactive app
│
├── notebooks/                  # Jupyter notebooks (to be created)
│   └── exploratory_analysis.ipynb
│
├── tests/                      # Unit tests (to be created)
│   └── __init__.py
│
├── data/                       # Data directory (created at runtime)
│   ├── raw/                   # Raw audio files and annotations
│   │   └── biodcase_development_set/
│   │       ├── train/
│   │       │   ├── audio/
│   │       │   └── annotations/
│   │       └── test/
│   │           ├── audio/
│   │           └── annotations/
│   └── processed/             # Processed data
│       ├── train/
│       │   ├── X.npy
│       │   ├── y.npy
│       │   └── label_mapping.pkl
│       └── test/
│           ├── X.npy
│           ├── y.npy
│           └── label_mapping.pkl
│
├── models/                     # Saved models (created at runtime)
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── lgbm_model.pkl
│   ├── nn_model.pkl
│   └── bayes_model.pkl
│
├── results/                    # Evaluation results (created at runtime)
│   ├── svm_results.pkl
│   ├── rf_results.pkl
│   └── ...
│
├── logs/                       # Log files (created at runtime)
│   └── training_YYYYMMDD_HHMMSS.log
│
└── visualizations/             # Generated plots (created at runtime)
    ├── svm_confusion_matrix.png
    ├── rf_confusion_matrix.png
    └── ...
```

## Module Descriptions

### `src/data/`

#### `preprocessing.py`
- **AudioPreprocessor**: Loads audio files and computes spectrograms
- **AnnotationParser**: Parses CSV annotation files
- **ImagePreprocessor**: Resizes and normalizes spectrogram images
- **DatasetBuilder**: Builds complete datasets from raw data
- **LabelEncoder**: Encodes and merges class labels

#### `augmentation.py`
- **AudioAugmenter**: Audio data augmentation (time stretch, pitch shift, noise)
- **ImageAugmenter**: Image augmentation for spectrograms

#### `dataset.py`
- **WhaleVocalizationDataset**: PyTorch Dataset class
- **BalancedBatchSampler**: Sampler for balanced batches

### `src/models/`

#### `svm_classifier.py`
- **SVMClassifier**: SVM with grid search and multiple kernels
- **MultiKernelSVM**: Ensemble of SVM with different kernels

#### `neural_network.py`
- **FeedForwardNN**: Fully connected neural network
- **NeuralNetworkClassifier**: Wrapper with training and evaluation

#### `bayesian_classifier.py`
- **BayesianClassifier**: Gaussian Naive Bayes
- **MultimodalBayesian**: Bayesian with Gaussian Mixture Models

#### `tree_models.py`
- **RandomForestModel**: Random Forest with grid search
- **XGBoostModel**: XGBoost classifier
- **LightGBMModel**: LightGBM classifier
- **DecisionTreeModel**: Decision tree with pruning

#### `clustering.py`
- **KMeansClusterer**: K-Means clustering
- **DBSCANClusterer**: DBSCAN clustering
- **HDBSCANClusterer**: Hierarchical DBSCAN
- **MeanShiftClusterer**: Mean Shift clustering
- **ClusteringEnsemble**: Ensemble of clustering methods

### `src/features/`

#### `feature_extraction.py`
- **FeatureExtractor**: Extracts statistical, spectral, temporal features
- **FeatureSelector**: Feature selection and dimensionality reduction

### `src/evaluation/`

#### `metrics.py`
- **ClassificationEvaluator**: Computes classification metrics
- **CrossValidationEvaluator**: Cross-validation evaluation
- **LearningCurveAnalyzer**: Analyzes learning curves

### `src/utils/`

#### `logger.py`
- **setup_logger**: Configures logging
- **MetricsLogger**: Logs training metrics to CSV

#### `visualization.py`
- **Visualizer**: Various plotting functions for analysis

## Scripts

### `scripts/prepare_data.py`
Prepares the dataset by:
1. Loading raw audio files
2. Computing spectrograms
3. Extracting annotated patches
4. Normalizing and resizing
5. Saving processed data

**Usage:**
```bash
python scripts/prepare_data.py --config config/config.yaml
```

### `scripts/train.py`
Trains classification models:
- Supports multiple model types
- Optional grid search for hyperparameters
- Feature extraction option
- Saves trained models and results

**Usage:**
```bash
python scripts/train.py --model svm --grid_search
```

### `scripts/evaluate.py`
Evaluates trained models:
- Computes test set metrics
- Generates confusion matrices
- Optional cross-validation

**Usage:**
```bash
python scripts/evaluate.py --model_type svm --model_path models/svm_model.pkl
```

### `scripts/inference.py`
Makes predictions on new audio files:
- Loads trained model
- Processes audio file
- Outputs prediction and probabilities

**Usage:**
```bash
python scripts/inference.py --model_type xgb --model_path models/xgb_model.pkl --audio_path audio.wav
```

## Web Application

### `app/streamlit_app.py`
Interactive Streamlit web application featuring:
- Upload and classify audio files
- View spectrograms
- Display prediction probabilities
- Explore model performance metrics
- Interactive visualizations

**Usage:**
```bash
streamlit run app/streamlit_app.py
```

## Configuration

### `config/config.yaml`
Central configuration file containing:
- Project settings
- Data paths and preprocessing parameters
- Spectrogram generation parameters
- Model hyperparameters
- Training settings
- Evaluation metrics
- Output directories

## Data Flow

```
Raw Audio Files
    ↓
[AudioPreprocessor]
    ↓
Spectrograms
    ↓
[Annotation-based Extraction]
    ↓
Image Patches
    ↓
[ImagePreprocessor]
    ↓
Normalized Images
    ↓
[FeatureExtractor] (optional)
    ↓
Feature Vectors
    ↓
[Model Training]
    ↓
Trained Models
    ↓
[Evaluation]
    ↓
Results & Metrics
```

## File Formats

- **Audio**: `.wav` files (250 Hz sampling rate)
- **Annotations**: `.csv` files with time-frequency coordinates
- **Processed Data**: `.npy` NumPy arrays
- **Models**: `.pkl` pickled models or `.pth` PyTorch models
- **Results**: `.pkl` pickled dictionaries
- **Visualizations**: `.png` images

## Key Features

1. **Modular Design**: Separate modules for data, models, features, evaluation
2. **Flexible Configuration**: YAML-based configuration system
3. **Multiple Models**: Implements 6+ classification algorithms
4. **Feature Engineering**: Comprehensive feature extraction
5. **Interactive App**: Streamlit web interface
6. **Comprehensive Evaluation**: Multiple metrics and visualizations
7. **Production Ready**: Logging, error handling, documentation

## Extension Points

To add new functionality:

1. **New Model**: Add to `src/models/` and update `scripts/train.py`
2. **New Features**: Extend `FeatureExtractor` in `src/features/`
3. **New Metrics**: Add to `ClassificationEvaluator` in `src/evaluation/`
4. **New Preprocessing**: Extend classes in `src/data/`

## Dependencies

See `requirements.txt` for complete list. Main dependencies:
- **Audio Processing**: librosa, soundfile
- **Image Processing**: opencv-python, Pillow
- **ML/DL**: scikit-learn, xgboost, lightgbm, torch
- **Clustering**: hdbscan
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: streamlit
- **Utilities**: numpy, pandas, scipy, pyyaml, tqdm

## Testing

Run the installation test:
```bash
python test_install.py
```

This verifies:
- All dependencies are installed
- Project structure is correct
- All modules can be imported
