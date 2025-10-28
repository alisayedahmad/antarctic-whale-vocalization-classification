#!/usr/bin/env python3
"""Inference script for whale vocalization classification."""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import AudioPreprocessor, ImagePreprocessor
from src.models.svm_classifier import SVMClassifier
from src.models.neural_network import NeuralNetworkClassifier
from src.models.bayesian_classifier import BayesianClassifier
from src.models.tree_models import RandomForestModel, XGBoostModel, LightGBMModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Inference for whale vocalization classification'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['svm', 'rf', 'xgb', 'lgbm', 'nn', 'bayes'],
        required=True,
        help='Type of model'
    )
    parser.add_argument(
        '--audio_path',
        type=str,
        required=True,
        help='Path to audio file'
    )
    parser.add_argument(
        '--output_proba',
        action='store_true',
        help='Output prediction probabilities'
    )
    
    return parser.parse_args()


def load_model(model_type, model_path, config):
    """Load trained model."""
    if model_type == 'svm':
        model = SVMClassifier(config)
    elif model_type == 'rf':
        model = RandomForestModel(config)
    elif model_type == 'xgb':
        model = XGBoostModel(config)
    elif model_type == 'lgbm':
        model = LightGBMModel(config)
    elif model_type == 'nn':
        model = NeuralNetworkClassifier(config)
    elif model_type == 'bayes':
        model = BayesianClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_model(model_path)
    return model


def main():
    """Main inference pipeline."""
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Antarctic Whale Vocalization - Inference")
    print("="*60)
    
    model = load_model(args.model_type, args.model_path, config)
    print(f"Loaded {args.model_type} model from {args.model_path}")
    
    audio_preprocessor = AudioPreprocessor(config)
    image_preprocessor = ImagePreprocessor(config)
    
    print(f"\nProcessing audio file: {args.audio_path}")
    audio, sr = audio_preprocessor.load_audio(args.audio_path)
    print(f"Audio duration: {len(audio)/sr:.2f} seconds")
    
    spec = audio_preprocessor.compute_spectrogram(audio)
    processed_spec = image_preprocessor.preprocess(spec)
    
    X = processed_spec.reshape(1, *processed_spec.shape)
    
    prediction = model.predict(X)[0]
    
    label_mapping = {
        0: 'ABZ (Antarctic Blue Whale - A/B/Z units)',
        1: 'DDswp (Downsweep - Blue/Fin whale D-calls)',
        2: '20Hz20Plus (Fin Whale - 20Hz pulses)'
    }
    
    predicted_class = label_mapping.get(prediction, f"Class {prediction}")
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Class: {predicted_class}")
    
    if args.output_proba and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        
        print("\nPrediction Probabilities:")
        for idx, prob in enumerate(proba):
            class_name = label_mapping.get(idx, f"Class {idx}")
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
