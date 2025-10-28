#!/usr/bin/env python3
"""Streamlit app for whale vocalization classification."""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import yaml
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import librosa
import librosa.display

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import AudioPreprocessor, ImagePreprocessor
from src.features.feature_extraction import FeatureExtractor
from src.models.svm_classifier import SVMClassifier
from src.models.neural_network import NeuralNetworkClassifier
from src.models.tree_models import RandomForestModel, XGBoostModel
from src.models.bayesian_classifier import BayesianClassifier


st.set_page_config(
    page_title="Antarctic Whale Vocalization Classifier",
    page_icon="🐋",
    layout="wide"
)


@st.cache_resource
def load_config():
    """Load configuration."""
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@st.cache_resource
def load_model(model_type, model_path, _config):
    """Load trained model."""
    if model_type == 'SVM':
        model = SVMClassifier(_config)
    elif model_type == 'Random Forest':
        model = RandomForestModel(_config)
    elif model_type == 'XGBoost':
        model = XGBoostModel(_config)
    elif model_type == 'Neural Network':
        model = NeuralNetworkClassifier(_config)
    elif model_type == 'Naive Bayes':
        model = BayesianClassifier(_config)
    else:
        return None
    
    try:
        model.load_model(model_path)
        return model
    except:
        return None


@st.cache_data
def load_results(results_path):
    """Load evaluation results."""
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        return results
    except:
        return None


def plot_spectrogram(audio, sr, config):
    """Plot spectrogram."""
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=config['data']['spectrogram']['nfft'],
        hop_length=config['data']['spectrogram']['hop_length'],
        win_length=config['data']['spectrogram']['win_length'],
        n_mels=config['data']['spectrogram']['n_mels']
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        spec_db,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        hop_length=config['data']['spectrogram']['hop_length'],
        ax=ax
    )
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    return fig


def main():
    """Main Streamlit app."""
    config = load_config()
    
    st.title("🐋 Antarctic Whale Vocalization Classifier")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "About"])
    
    with tab1:
        st.header("Make Predictions")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            model_type = st.selectbox(
                "Select Model",
                ['SVM', 'Random Forest', 'XGBoost', 'Neural Network', 'Naive Bayes']
            )
            
            models_dir = Path(config['output']['models_dir'])
            model_map = {
                'SVM': 'svm_model.pkl',
                'Random Forest': 'rf_model.pkl',
                'XGBoost': 'xgb_model.pkl',
                'Neural Network': 'nn_model.pkl',
                'Naive Bayes': 'bayes_model.pkl'
            }
            
            model_path = models_dir / model_map[model_type]
            
            if model_path.exists():
                model = load_model(model_type, str(model_path), config)
                
                if model:
                    st.success(f"{model_type} model loaded successfully!")
                else:
                    st.error("Failed to load model")
            else:
                st.warning(f"Model not found at {model_path}")
                st.info("Please train the model first using: `python scripts/train.py`")
        
        with col2:
            st.info("Upload an audio file (.wav) containing whale vocalizations")
        
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
        
        if uploaded_file and model:
            st.markdown("---")
            
            audio_preprocessor = AudioPreprocessor(config)
            image_preprocessor = ImagePreprocessor(config)
            
            audio, sr = librosa.load(uploaded_file, sr=config['data']['spectrogram']['sample_rate'])
            
            st.subheader("Audio Waveform")
            fig_wave, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(audio, sr=sr, ax=ax)
            ax.set_title('Waveform')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            st.pyplot(fig_wave)
            
            st.subheader("Spectrogram")
            fig_spec = plot_spectrogram(audio, sr, config)
            st.pyplot(fig_spec)
            
            spec = audio_preprocessor.compute_spectrogram(audio)
            processed_spec = image_preprocessor.preprocess(spec)
            
            if st.button("Classify Vocalization", type="primary"):
                with st.spinner("Analyzing..."):
                    X = processed_spec.reshape(1, *processed_spec.shape)
                    
                    prediction = model.predict(X)[0]
                    
                    label_mapping = {
                        0: 'ABZ (Antarctic Blue Whale)',
                        1: 'DDswp (Downsweep)',
                        2: '20Hz20Plus (Fin Whale)'
                    }
                    
                    predicted_class = label_mapping.get(prediction, f"Class {prediction}")
                    
                    st.success(f"**Predicted Class: {predicted_class}**")
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)[0]
                        
                        st.subheader("Prediction Probabilities")
                        prob_df = {
                            'Class': [label_mapping.get(i, f"Class {i}") for i in range(len(proba))],
                            'Probability': proba
                        }
                        
                        fig_prob, ax = plt.subplots(figsize=(8, 4))
                        ax.barh(prob_df['Class'], prob_df['Probability'])
                        ax.set_xlabel('Probability')
                        ax.set_xlim([0, 1])
                        plt.tight_layout()
                        st.pyplot(fig_prob)
    
    with tab2:
        st.header("Model Performance")
        
        model_select = st.selectbox(
            "Select model to view performance",
            ['SVM', 'Random Forest', 'XGBoost', 'Neural Network', 'Naive Bayes'],
            key='model_perf'
        )
        
        results_dir = Path(config['output']['results_dir'])
        model_results_map = {
            'SVM': 'svm_results.pkl',
            'Random Forest': 'rf_results.pkl',
            'XGBoost': 'xgb_results.pkl',
            'Neural Network': 'nn_results.pkl',
            'Naive Bayes': 'bayes_results.pkl'
        }
        
        results_path = results_dir / model_results_map[model_select]
        
        if results_path.exists():
            results = load_results(str(results_path))
            
            if results:
                st.subheader("Overall Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{results['metrics']['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{results['metrics']['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{results['metrics']['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{results['metrics']['f1_score']:.3f}")
                
                st.subheader("Confusion Matrix")
                
                viz_dir = Path(config['output']['visualizations_dir'])
                cm_path = viz_dir / f"{model_results_map[model_select].replace('_results.pkl', '_confusion_matrix.png')}"
                
                if cm_path.exists():
                    cm_img = Image.open(cm_path)
                    st.image(cm_img, use_container_width=True)
                
                st.subheader("Per-Class Metrics")
                per_class = results['per_class_metrics']
                
                metrics_data = []
                for class_name, metrics in per_class.items():
                    if isinstance(metrics, dict) and 'precision' in metrics:
                        metrics_data.append({
                            'Class': class_name,
                            'Precision': f"{metrics['precision']:.3f}",
                            'Recall': f"{metrics['recall']:.3f}",
                            'F1-Score': f"{metrics['f1-score']:.3f}",
                            'Support': metrics['support']
                        })
                
                if metrics_data:
                    st.table(metrics_data)
        else:
            st.warning(f"Results not found for {model_select}")
            st.info("Train the model first using: `python scripts/train.py`")
    
    with tab3:
        st.header("About This Project")
        
        st.markdown("""
        ### Antarctic Whale Vocalization Classification
        
        This application classifies whale vocalizations from Antarctic blue whales and fin whales
        using machine learning techniques.
        
        **Classification Categories:**
        - **ABZ**: Antarctic blue whale calls (A, B, Z units)
        - **DDswp**: Downsweep calls (blue and fin whale D-calls)
        - **20Hz20Plus**: Fin whale 20 Hz pulses
        
        **Features:**
        - Multiple classification algorithms (SVM, Random Forest, XGBoost, Neural Networks, Naive Bayes)
        - Real-time audio analysis and prediction
        - Comprehensive performance metrics
        - Interactive visualizations
        
        **Dataset:**
        - 6,591 audio recordings from Antarctic deployments (2005-2017)
        - 1,880 hours of underwater acoustic data
        - 75,454 annotated whale vocalizations
        
        **Academic Context:**
        Machine Learning Course Project - 2nd Year Engineering School
        
        ---
        
        **How to Use:**
        1. Select a trained model from the dropdown
        2. Upload a .wav audio file containing whale vocalizations
        3. View the spectrogram and click "Classify Vocalization"
        4. Explore model performance metrics in the "Model Performance" tab
        """)
        
        st.markdown("### References")
        st.markdown("""
        - Rankin et al. (2005). Vocalisations of Antarctic blue whales
        - Širović et al. (2004). Seasonality of blue and fin whale calls
        - Gedamke & Robinson (2010). Acoustic survey for marine mammal occurrence
        """)


if __name__ == '__main__':
    main()
