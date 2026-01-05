"""
Milestone 3: Visualization of Model Results
This module creates confusion matrix and ROC-AUC visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os


class ModelVisualizer:
    
    def __init__(self, results_dict, y_test, output_dir='visualizations'):
       
        self.results = results_dict
        self.y_test = y_test
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """Plot confusion matrix for a single model."""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results.")
            return None
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=np.unique(self.y_test),
                    yticklabels=np.unique(self.y_test))
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {filepath}")
        
        plt.close()
        return filepath
    
    def plot_all_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        print("\n" + "="*60)
        print("GENERATING CONFUSION MATRICES")
        print("="*60)
        
        filepaths = []
        for model_name in self.results.keys():
            filepath = self.plot_confusion_matrix(model_name)
            if filepath:
                filepaths.append(filepath)
        
        return filepaths
    
    def plot_roc_curve(self, model_name, figsize=(8, 6)):
        """Plot ROC curve for a single model."""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results.")
            return None
        
        results = self.results[model_name]
        y_pred_proba = results['y_pred_proba']
        
        if y_pred_proba is None:
            print(f"No probability predictions available for {model_name}")
            return None
        
        try:
            plt.figure(figsize=figsize)
            
            # Handle binary vs multi-class
            if len(np.unique(self.y_test)) == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
            else:
                # Multi-class classification - plot per class
                unique_classes = np.unique(self.y_test)
                for i, class_label in enumerate(unique_classes):
                    y_test_binary = (self.y_test == class_label).astype(int)
                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC-AUC Curve - {model_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            filepath = os.path.join(self.output_dir, f'roc_curve_{model_name}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {filepath}")
            
            plt.close()
            return filepath
            
        except Exception as e:
            print(f"Error plotting ROC curve for {model_name}: {str(e)}")
            return None
    
    def plot_all_roc_curves(self):
        """Plot ROC curves for all models."""
        print("\n" + "="*60)
        print("GENERATING ROC-AUC CURVES")
        print("="*60)
        
        filepaths = []
        for model_name in self.results.keys():
            filepath = self.plot_roc_curve(model_name)
            if filepath:
                filepaths.append(filepath)
        
        return filepaths
    
    def plot_metrics_comparison(self, figsize=(12, 6)):
        """Plot comparison of metrics across all models."""
        print("\n" + "="*60)
        print("GENERATING METRICS COMPARISON PLOT")
        print("="*60)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        model_names = list(self.results.keys())
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for model_name in model_names:
            for metric in metrics:
                data[metric].append(self.results[model_name].get(metric, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            offset = (i - 2) * width
            ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in model_names])
        ax.legend(loc='best', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {filepath}")
        
        plt.close()
        return filepath
    
    def plot_combined_confusion_matrices(self, figsize=(15, 10)):
        """Plot all confusion matrices in a single figure."""
        print("\n" + "="*60)
        print("GENERATING COMBINED CONFUSION MATRICES")
        print("="*60)
        
        num_models = len(self.results)
        if num_models == 0:
            print("No models to plot.")
            return None
        
        # Calculate grid layout
        rows = (num_models + 1) // 2
        cols = 2 if num_models > 1 else 1
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, ax) in enumerate(zip(self.results.keys(), axes)):
            cm = self.results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       ax=ax, xticklabels=np.unique(self.y_test),
                       yticklabels=np.unique(self.y_test))
            ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Hide extra subplots
        for idx in range(num_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, 'combined_confusion_matrices.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Combined confusion matrices saved to {filepath}")
        
        plt.close()
        return filepath
    
    def plot_combined_roc_curves(self, figsize=(10, 8)):
        """Plot all ROC curves in a single figure."""
        print("\n" + "="*60)
        print("GENERATING COMBINED ROC CURVES")
        print("="*60)
        
        if not self.results:
            print("No models to plot.")
            return None
        
        plt.figure(figsize=figsize)
        
        for model_name in self.results.keys():
            results = self.results[model_name]
            y_pred_proba = results['y_pred_proba']
            
            if y_pred_proba is None:
                continue
            
            try:
                if len(np.unique(self.y_test)) == 2:
                    fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, 
                            label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Warning: Could not plot ROC for {model_name}: {str(e)}")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC-AUC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(self.output_dir, 'combined_roc_curves.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Combined ROC curves saved to {filepath}")
        
        plt.close()
        return filepath
    
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*60)
        
        filepaths = []
        filepaths.append(self.plot_all_confusion_matrices())
        filepaths.append(self.plot_all_roc_curves())
        filepaths.append(self.plot_metrics_comparison())
        filepaths.append(self.plot_combined_confusion_matrices())
        filepaths.append(self.plot_combined_roc_curves())
        
        print("\n" + "="*60)
        print(f"All visualizations saved to '{self.output_dir}' directory")
        print("="*60)
        
        return filepaths


if __name__ == "__main__":
    print("Model Visualization Module - For use with model_training.py and model_evaluation.py")
    print("See model_training.py for usage examples")
