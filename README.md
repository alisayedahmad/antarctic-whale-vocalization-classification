# Antarctic Whale Vocalization Classification

Multi-class classification system for Antarctic blue whale and fin whale vocalizations using machine learning techniques.

## Project Overview

This project implements a supervised multi-class classification pipeline for identifying and classifying whale vocalizations from spectrogram images. Developed as part of a Machine Learning course during the second year of engineering school, this system addresses the challenging task of distinguishing between different call types produced by Antarctic blue whales and fin whales in underwater acoustic recordings.

### Classification Categories

The system identifies 7 distinct vocalization types, which are grouped into 3 main categories for evaluation:

**Antarctic Blue Whale (BmAnt):**
- **BmZ**: Z-call with three components (A, B, C) - smooth frequency transition from 27 to 16 Hz
- **BmA**: A-unit only
- **BmB**: A and B units
- **BmD**: D-call with downsweeping frequency component (20–120 Hz)

**Fin Whale (Bp):**
- **Bp20**: 20 Hz pulse without overtone (30–15 Hz)
- **Bp20Plus**: 20 Hz pulse with secondary energy (80–120 Hz variation)
- **BpD**: 40 Hz downsweep (30–90 Hz)

## Dataset

The project uses the bioDCASE development dataset comprising:
- **Training set**: 6,007 audio files (1,292 hours, 57,827 events) from 8 site-year deployments
- **Test set**: 587 audio files (585 hours, 17,627 events) from 3 site-year deployments
- **Recording period**: 2005–2017 across Antarctic deployment sites
- **Sampling rate**: 250 Hz
- **Format**: WAV audio files with CSV annotations

## Methods Implemented

### Classification Algorithms
- **Support Vector Machines (SVM)** with RBF and linear kernels
- **Neural Networks** (fully connected architectures, no CNNs)
- **Bayesian Classifier** (Naive Bayes variants)
- **Decision Trees and Ensembles**:
  - Random Forest
  - Gradient Boosted Trees (XGBoost, LightGBM)
  - Pruning techniques

### Clustering Methods
- K-Means
- DBSCAN (Density-Based Spatial Clustering)
- HDBSCAN (Hierarchical DBSCAN)
- Mean Shift

## Project Structure

```
antarctic-whale-vocalization-classification/
├── config/                 # Configuration files
├── src/
│   ├── data/              # Data preprocessing and loading
│   ├── models/            # ML model implementations
│   ├── features/          # Feature extraction
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # Utilities and visualization
├── scripts/               # Training and inference scripts
├── notebooks/             # Exploratory analysis
├── app/                   # Streamlit web application
└── tests/                 # Unit tests
```

## Installation

```bash
git clone https://github.com/yourusername/antarctic-whale-vocalization-classification.git
cd antarctic-whale-vocalization-classification
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

```bash
python scripts/prepare_data.py --data_path /path/to/biodcase_development_set
```

### 2. Training Models

```bash
# Train all models
python scripts/train.py --config config/config.yaml

# Train specific model
python scripts/train.py --model svm --config config/config.yaml
```

### 3. Evaluation

```bash
python scripts/evaluate.py --model_path models/best_model.pkl --test_data data/processed/test
```

### 4. Interactive Demo

```bash
streamlit run app/streamlit_app.py
```

## Pipeline

1. **Spectrogram Generation**: Audio files are transformed into spectrograms using Short-Time Fourier Transform (STFT)
   - NFFT: 512
   - Window size: 256
   - Overlap: 98%
2. **Image Extraction**: Vocalizations are extracted as image patches from spectrograms based on annotation coordinates (time-frequency bounding boxes)
3. **Feature Engineering**: Multiple feature sets extracted including statistical, spectral, and temporal features
4. **Classification**: Trained models predict the vocalization class from input spectrograms
5. **Evaluation**: Performance measured using precision, recall, F1-score, and confusion matrices

## Results

Model performance on the test set:

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| SVM | **0.86** | **0.84** | 0.85 | 0.83 |
| Random Forest | 0.82 | 0.80 | 0.81 | 0.79 |
| XGBoost | **0.88** | **0.86** | 0.87 | 0.85 |
| Neural Network | 0.84 | 0.82 | 0.83 | 0.81 |

**Per-class F1 (XGBoost, test):**

- **BmZ:** 0.92
- **BmA:** 0.81
- **BmB:** 0.85
- **BmD:** 0.74
- **Bp20:** 0.90
- **Bp20Plus:** 0.79
- **BpD:** 0.76

**Grouped category accuracy (test):**
- **Blue whale (BmAnt overall):** 0.87
- **Fin whale (Bp overall):** 0.90

**Common confusions:**
- **BmD ↔ BpD:** both downsweeps with overlapping frequency bands
- **Bp20 ↔ Bp20Plus:** secondary energy component sometimes faint
- **BmA ↔ BmB:** transition between A- and B-units unclear in noisy samples

## Technical Challenges

- **Low-frequency vocalizations**: Calls below 120 Hz, often masked by background noise
- **Temporal and spectral overlap**: Different call types overlap in time and frequency
- **Variable image sizes**: Extracted spectrogram patches differ per annotation
- **Class imbalance**: Some call types far more common (e.g., BmZ vs BpD)
- **Ambiguous calls**: D-calls (BmD vs BpD) visually similar in spectrograms

## References

- Rankin et al. (2005). *Vocalisations of Antarctic blue whales*
- Širović et al. (2004). *Seasonality of blue and fin whale calls*
- Gedamke & Robinson (2010). *Acoustic survey for marine mammal occurrence*
- Miller et al. (2014). *Blue whale vocalizations recorded around New Zealand*

## License

This project is licensed under the **MIT License**.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

**Academic Context:** Machine Learning Course Project — 2nd Year Engineering School
