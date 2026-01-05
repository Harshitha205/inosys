"""
Milestone 3: Model Evaluation
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


class ModelEvaluator:
    
    def __init__(self, X_test, y_test, models):
        self.X_test = X_test
        self.y_test = y_test
        self.models = models
        self.results = {}
    
    def evaluate_all(self):
        print("\n=== Model Evaluation ===")
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted'),
                'recall': recall_score(self.y_test, y_pred, average='weighted'),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(self.y_test, y_proba),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            print(f"\n{name}:")
            print(f"  Accuracy: {self.results[name]['accuracy']:.4f}")
            print(f"  F1-Score: {self.results[name]['f1_score']:.4f}")
            print(f"  ROC-AUC:  {self.results[name]['roc_auc']:.4f}")
        
        return self.results
    
    def get_comparison_table(self):
        data = [{
            'Model': name.replace('_', ' ').title(),
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1_score']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}"
        } for name, m in self.results.items()]
        return pd.DataFrame(data)
    
    def get_best_model(self, metric='f1_score'):
        best = max(self.results.items(), key=lambda x: x[1][metric])
        print(f"\n✓ Best Model: {best[0]} ({metric}: {best[1][metric]:.4f})")
        return best[0], self.models[best[0]]
