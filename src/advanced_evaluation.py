"""
Advanced model evaluation and visualization utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os
from typing import List, Dict, Any
from datetime import datetime

class ModelEvaluator:
    """Advanced model evaluation with visualizations."""
    
    def __init__(self, model, vectorizer, label_encoder, save_dir="evaluation_results"):
        self.model = model
        self.vectorizer = vectorizer  
        self.label_encoder = label_encoder
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def evaluate_comprehensive(self, X_test, y_test, save_plots=True):
        """Comprehensive model evaluation with visualizations."""
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Basic metrics
        results = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        if save_plots:
            # Generate visualizations
            self._plot_confusion_matrix(y_test, y_pred)
            self._plot_classification_report(y_test, y_pred)
            self._plot_prediction_confidence(y_pred_proba)
            self._plot_roc_curves(y_test, y_pred_proba)
            
        # Save results
        self._save_results(results)
        
        return results
    
    def _calculate_metrics(self, y_test, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        results['per_class_metrics'] = report
        
        # Confidence statistics
        max_probas = np.max(y_pred_proba, axis=1)
        results['confidence_stats'] = {
            'mean_confidence': float(np.mean(max_probas)),
            'std_confidence': float(np.std(max_probas)),
            'min_confidence': float(np.min(max_probas)),
            'max_confidence': float(np.max(max_probas)),
            'high_confidence_ratio': float(np.mean(max_probas > 0.8))
        }
        
        return results
    
    def _plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(y_test, y_pred)
        class_names = self.label_encoder.classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        plt.savefig(f'{self.save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_classification_report(self, y_test, y_pred):
        """Plot classification report as heatmap."""
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
        df = df.drop(['support'], axis=1)  # Remove support column for cleaner plot
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0.5, 
                   fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Classification Report Heatmap')
        plt.tight_layout()
        
        plt.savefig(f'{self.save_dir}/classification_report.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_confidence(self, y_pred_proba):
        """Plot prediction confidence distribution."""
        max_probas = np.max(y_pred_proba, axis=1)
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(max_probas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Maximum Prediction Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.axvline(np.mean(max_probas), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probas):.3f}')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(max_probas)
        plt.ylabel('Maximum Prediction Probability')
        plt.title('Confidence Box Plot')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/prediction_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, y_test, y_pred_proba):
        """Plot ROC curves for multi-class classification.""" 
        try:
            plt.figure(figsize=(12, 8))
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            # For each class, calculate ROC curve using one-vs-rest approach
            for i, class_name in enumerate(self.label_encoder.classes_):
                # Create binary labels for this class vs all others
                y_binary = (y_test == i).astype(int)
                
                if len(np.unique(y_binary)) > 1:  # Check if both classes exist
                    fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    color = colors[i % len(colors)]
                    plt.plot(fpr, tpr, color=color, lw=2,
                            label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate') 
            plt.title('Multi-Class ROC Curves (One-vs-Rest)')
            plt.legend(loc="lower right")
            
            plt.savefig(f'{self.save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate ROC curves: {e}")
            # Create a simple placeholder plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'ROC Curves not available\nfor this dataset', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('ROC Curves')
            plt.savefig(f'{self.save_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_results(self, results):
        """Save evaluation results to JSON."""
        with open(f'{self.save_dir}/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def generate_report(self, results):
        """Generate a comprehensive evaluation report."""
        report = f"""
# Model Evaluation Report

**Generated on:** {results['timestamp']}

## Overall Performance
- **Accuracy:** {results['accuracy']:.4f}
- **F1-Score (Macro):** {results['f1_macro']:.4f}
- **F1-Score (Weighted):** {results['f1_weighted']:.4f}
- **Precision (Macro):** {results['precision_macro']:.4f}
- **Recall (Macro):** {results['recall_macro']:.4f}

## Confidence Analysis
- **Mean Confidence:** {results['confidence_stats']['mean_confidence']:.4f}
- **High Confidence Predictions (>0.8):** {results['confidence_stats']['high_confidence_ratio']:.2%}
- **Confidence Range:** {results['confidence_stats']['min_confidence']:.4f} - {results['confidence_stats']['max_confidence']:.4f}

## Per-Class Performance
"""
        
        # Add per-class metrics table
        for class_name, metrics in results['per_class_metrics'].items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                report += f"- **{class_name}:** Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}\n"
        
        # Save report
        with open(f'{self.save_dir}/evaluation_report.md', 'w') as f:
            f.write(report)
        
        return report
