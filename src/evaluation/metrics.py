import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ClassificationEvaluator:
    """Evaluates classification models."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict:
        """Compute classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        return metrics
    
    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None
    ) -> Dict:
        """Compute per-class metrics."""
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        save_path: str = None,
        normalize: bool = False
    ):
        """Plot confusion matrix."""
        cm = self.compute_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str] = None
    ) -> Dict:
        """Comprehensive model evaluation."""
        y_pred = model.predict(X_test)
        
        metrics = self.compute_metrics(y_test, y_pred)
        per_class_metrics = self.compute_per_class_metrics(y_test, y_pred, class_names)
        cm = self.compute_confusion_matrix(y_test, y_pred)
        
        results = {
            'metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        return results
    
    def compare_models(
        self,
        results: Dict[str, Dict],
        metric: str = 'f1_score'
    ):
        """Compare multiple models."""
        model_names = list(results.keys())
        scores = [results[name]['metrics'][metric] for name in model_names]
        
        plt.figure(figsize=(12, 6))
        plt.bar(model_names, scores)
        plt.title(f'Model Comparison - {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def print_results(self, results: Dict, model_name: str = "Model"):
        """Print evaluation results."""
        print(f"\n{'='*60}")
        print(f"{model_name} Evaluation Results")
        print(f"{'='*60}")
        
        print("\nOverall Metrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        print("\nPer-Class Metrics:")
        per_class = results['per_class_metrics']
        for class_name, metrics in per_class.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"\n  {class_name}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-Score: {metrics['f1-score']:.4f}")
                print(f"    Support: {metrics['support']}")


class CrossValidationEvaluator:
    """Performs cross-validation evaluation."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def cross_validate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score, cross_validate
        
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        }
        
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        results = {}
        for metric in scoring.keys():
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'
            
            results[metric] = {
                'test_mean': cv_results[test_key].mean(),
                'test_std': cv_results[test_key].std(),
                'train_mean': cv_results[train_key].mean(),
                'train_std': cv_results[train_key].std()
            }
        
        return results
    
    def print_cv_results(self, results: Dict):
        """Print cross-validation results."""
        print("\nCross-Validation Results:")
        print(f"{'Metric':<15} {'Test Mean':<12} {'Test Std':<12} {'Train Mean':<12} {'Train Std':<12}")
        print("-" * 63)
        
        for metric, values in results.items():
            print(f"{metric:<15} {values['test_mean']:<12.4f} {values['test_std']:<12.4f} "
                  f"{values['train_mean']:<12.4f} {values['train_std']:<12.4f}")


class LearningCurveAnalyzer:
    """Analyzes learning curves."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        save_path: str = None
    ):
        """Plot learning curve."""
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', label='Training score')
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1
        )
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation score')
        plt.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.1
        )
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
