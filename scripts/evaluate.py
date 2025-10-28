#!/usr/bin/env python3
"""Evaluation script for whale vocalization classification models."""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DatasetBuilder
from src.evaluation.metrics import ClassificationEvaluator, CrossValidationEvaluator
from src.models.svm_classifier import SVMClassifier
from src.models.neural_network import NeuralNetworkClassifier
from src.models.bayesian_classifier import BayesianClassifier
from src.models.tree_models import RandomForestModel, XGBoostModel, LightGBMModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate whale vocalization classification models'
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
        help='Path to trained model file'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['svm', 'rf', 'xgb', 'lgbm', 'nn', 'bayes'],
        required=True,
        help='Type of model to evaluate'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--cross_validate',
        action='store_true',
        help='Perform cross-validation'
    )
    
    return parser.parse_args()


def load_model(model_type, model_path, config):
    """Load trained model."""
    print(f"Loading {model_type} model from {model_path}...")
    
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
    """Main evaluation pipeline."""
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Antarctic Whale Vocalization - Model Evaluation")
    print("="*60)
    
    model = load_model(args.model_type, args.model_path, config)
    
    dataset_builder = DatasetBuilder(config)
    X_test, y_test, label_mapping = dataset_builder.load_processed_data('test')
    
    print(f"\nTest data shape: {X_test.shape}")
    print(f"Number of classes: {len(label_mapping)}")
    
    evaluator = ClassificationEvaluator(config)
    
    class_names = [k for k, v in sorted(label_mapping.items(), key=lambda x: x[1])]
    results = evaluator.evaluate_model(model, X_test, y_test, class_names)
    
    evaluator.print_results(results, model_name=args.model_type.upper())
    
    viz_dir = Path(config['output']['visualizations_dir'])
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = viz_dir / f'{args.model_type}_eval_confusion_matrix.png'
    evaluator.plot_confusion_matrix(
        y_test,
        results['predictions'],
        class_names,
        save_path=str(cm_path),
        normalize=True
    )
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    if args.cross_validate:
        print("\nPerforming cross-validation...")
        cv_evaluator = CrossValidationEvaluator(config)
        
        X_train, y_train, _ = dataset_builder.load_processed_data('train')
        
        cv_results = cv_evaluator.cross_validate_model(
            model,
            X_train,
            y_train,
            cv=config['training']['cross_validation']['n_folds']
        )
        
        cv_evaluator.print_cv_results(cv_results)
    
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / f'{args.model_type}_eval_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "="*60)
    print("Evaluation completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
