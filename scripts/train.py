#!/usr/bin/env python3
"""Training script for whale vocalization classification models."""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DatasetBuilder
from src.features.feature_extraction import FeatureExtractor
from src.models.svm_classifier import SVMClassifier
from src.models.neural_network import NeuralNetworkClassifier
from src.models.bayesian_classifier import BayesianClassifier
from src.models.tree_models import (
    RandomForestModel,
    XGBoostModel,
    LightGBMModel
)
from src.evaluation.metrics import ClassificationEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train whale vocalization classification models'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['svm', 'rf', 'xgb', 'lgbm', 'nn', 'bayes', 'all'],
        default='all',
        help='Model to train'
    )
    parser.add_argument(
        '--use_features',
        action='store_true',
        help='Extract and use engineered features'
    )
    parser.add_argument(
        '--grid_search',
        action='store_true',
        help='Use grid search for hyperparameter tuning'
    )
    
    return parser.parse_args()


def load_data(config):
    """Load processed data."""
    dataset_builder = DatasetBuilder(config)
    
    X_train, y_train, label_mapping = dataset_builder.load_processed_data('train')
    X_test, y_test, _ = dataset_builder.load_processed_data('test')
    
    return X_train, y_train, X_test, y_test, label_mapping


def extract_features(config, X_train, X_test):
    """Extract engineered features from images."""
    print("\nExtracting features...")
    feature_extractor = FeatureExtractor(config)
    
    X_train_features = feature_extractor.extract_batch_features(X_train)
    X_test_features = feature_extractor.extract_batch_features(X_test)
    
    print(f"Feature shapes: Train={X_train_features.shape}, Test={X_test_features.shape}")
    
    return X_train_features, X_test_features


def train_model(model_type, config, X_train, y_train, X_val, y_val, use_grid_search):
    """Train a specific model."""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model...")
    print(f"{'='*60}")
    
    if model_type == 'svm':
        model = SVMClassifier(config)
        model.train(X_train, y_train, use_grid_search=use_grid_search)
    
    elif model_type == 'rf':
        model = RandomForestModel(config)
        model.train(X_train, y_train, use_grid_search=use_grid_search)
    
    elif model_type == 'xgb':
        model = XGBoostModel(config)
        model.train(X_train, y_train, X_val, y_val)
    
    elif model_type == 'lgbm':
        model = LightGBMModel(config)
        model.train(X_train, y_train, X_val, y_val)
    
    elif model_type == 'nn':
        model = NeuralNetworkClassifier(config)
        model.train(X_train, y_train, X_val, y_val)
    
    elif model_type == 'bayes':
        model = BayesianClassifier(config)
        model.train(X_train, y_train)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def evaluate_and_save(model, model_type, config, X_test, y_test, label_mapping):
    """Evaluate model and save results."""
    evaluator = ClassificationEvaluator(config)
    
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]
    results = evaluator.evaluate_model(model, X_test, y_test, class_names)
    
    evaluator.print_results(results, model_name=model_type.upper())
    
    models_dir = Path(config['output']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f'{model_type}_model.pkl'
    model.save_model(str(model_path))
    
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / f'{model_type}_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    viz_dir = Path(config['output']['visualizations_dir'])
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = viz_dir / f'{model_type}_confusion_matrix.png'
    evaluator.plot_confusion_matrix(
        y_test,
        results['predictions'],
        class_names,
        save_path=str(cm_path),
        normalize=True
    )
    
    return results


def main():
    """Main training pipeline."""
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Antarctic Whale Vocalization - Model Training")
    print("="*60)
    
    X_train, y_train, X_test, y_test, label_mapping = load_data(config)
    
    print(f"\nData loaded:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Classes: {len(label_mapping)}")
    
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config['training']['validation_size'],
        random_state=config['project']['random_seed'],
        stratify=y_train
    )
    
    if args.use_features:
        X_train_full, X_val = extract_features(config, X_train_full, X_val)
        X_test_feat, _ = extract_features(config, X_test, X_test)
        X_test = X_test_feat
    
    models_to_train = ['svm', 'rf', 'xgb', 'lgbm', 'nn', 'bayes'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_type in models_to_train:
        try:
            model = train_model(
                model_type,
                config,
                X_train_full,
                y_train_full,
                X_val,
                y_val,
                args.grid_search
            )
            
            results = evaluate_and_save(
                model,
                model_type,
                config,
                X_test,
                y_test,
                label_mapping
            )
            
            all_results[model_type] = results
            
        except Exception as e:
            print(f"\nError training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}")
        print(f"\n{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        for model_name, results in all_results.items():
            metrics = results['metrics']
            print(f"{model_name:<10} {metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
