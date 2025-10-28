import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


class Visualizer:
    """Visualization utilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        sns.set_style('whitegrid')
        
    def plot_spectrograms(
        self,
        spectrograms: List[np.ndarray],
        labels: List[str],
        save_path: str = None
    ):
        """Plot multiple spectrograms."""
        n_specs = len(spectrograms)
        fig, axes = plt.subplots(1, n_specs, figsize=(5*n_specs, 4))
        
        if n_specs == 1:
            axes = [axes]
        
        for i, (spec, label) in enumerate(zip(spectrograms, labels)):
            im = axes[i].imshow(
                spec,
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            axes[i].set_title(label)
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Frequency')
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_distribution(
        self,
        y: np.ndarray,
        class_names: List[str],
        save_path: str = None
    ):
        """Plot class distribution."""
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar([class_names[i] for i in unique], counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=45, ha='right')
        
        for i, (cls, count) in enumerate(zip(unique, counts)):
            plt.text(i, count, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_importance: np.ndarray,
        feature_names: List[str] = None,
        top_n: int = 20,
        save_path: str = None
    ):
        """Plot feature importance."""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
        
        indices = np.argsort(feature_importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            [feature_names[i] for i in indices],
            feature_importance[indices]
        )
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: str = None
    ):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history['train_acc'], label='Train Accuracy')
        axes[1].plot(history['val_acc'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        save_path: str = None
    ):
        """Plot ROC curves for multi-class classification."""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr,
                tpr,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str],
        save_path: str = None
    ):
        """Plot precision-recall curves."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize
        
        n_classes = len(class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i],
                y_proba[:, i]
            )
            ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
            
            plt.plot(
                recall,
                precision,
                label=f'{class_names[i]} (AP = {ap:.2f})'
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
