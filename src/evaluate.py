"""
Model evaluation module for research abstract classification.

This module provides comprehensive evaluation of trained models including
accuracy, F1-score, precision, recall, and confusion matrix analysis.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple

try:
    from .preprocessing import TextPreprocessor, load_and_preprocess_data
except ImportError:
    from preprocessing import TextPreprocessor, load_and_preprocess_data


class ModelEvaluator:
    """
    Comprehensive model evaluation class for research abstract classification.
    
    This class provides detailed performance analysis including metrics calculation,
    visualization, and reporting capabilities.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model evaluator.
        
        Args:
            models_dir (str): Directory containing saved models
        """
        self.models_dir = models_dir
        self.classifier = None
        self.label_encoder = None
        self.preprocessor = TextPreprocessor()
        self.is_loaded = False
        
    def load_models(self) -> 'ModelEvaluator':
        """
        Load trained models from disk.
        
        Returns:
            ModelEvaluator: Self for method chaining
        """
        # Load classifier
        classifier_path = os.path.join(self.models_dir, "classifier.pkl")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
        self.classifier = joblib.load(classifier_path)
        
        # Load label encoder
        encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
        self.label_encoder = joblib.load(encoder_path)
        
        # Load vectorizer
        vectorizer_path = os.path.join(self.models_dir, "vectorizer.pkl")
        self.preprocessor.load_vectorizer(vectorizer_path)
        
        self.is_loaded = True
        print("Models loaded successfully!")
        
        return self
    
    def evaluate_on_test_data(self, data_path: str, test_size: float = 0.2, 
                             random_state: int = 42) -> Dict[str, Any]:
        """
        Evaluate model performance on test data split from the dataset.
        
        Args:
            data_path (str): Path to the CSV data file
            test_size (float): Fraction of data to use for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        print("Evaluating model on test data...")
        
        # Load and split data
        texts, labels = load_and_preprocess_data(data_path)
        _, test_texts, _, test_labels = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Prepare test features
        X_test = self.preprocessor.transform_texts(test_texts)
        y_test = self.label_encoder.transform(test_labels)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        results['test_texts'] = test_texts
        results['true_labels'] = test_labels
        results['predicted_labels'] = self.label_encoder.inverse_transform(y_pred)
        
        return results
    
    def evaluate_on_custom_data(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate model performance on custom text data.
        
        Args:
            texts (List[str]): List of text samples
            labels (List[str]): List of true labels
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        print(f"Evaluating model on {len(texts)} custom samples...")
        
        # Prepare features
        X = self.preprocessor.transform_texts(texts)
        y_true = self.label_encoder.transform(labels)
        
        # Make predictions
        y_pred = self.classifier.predict(X)
        y_pred_proba = self.classifier.predict_proba(X)
        
        # Calculate metrics
        results = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        results['test_texts'] = texts
        results['true_labels'] = labels
        results['predicted_labels'] = self.label_encoder.inverse_transform(y_pred)
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Prediction probabilities
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        target_names = self.label_encoder.classes_
        class_report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Confidence scores
        max_proba = np.max(y_pred_proba, axis=1)
        mean_confidence = np.mean(max_proba)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'mean_confidence': mean_confidence,
            'class_names': target_names,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """
        Print a comprehensive evaluation report.
        
        Args:
            results (Dict[str, Any]): Evaluation results from evaluate methods
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"Precision (Macro): {results['precision_macro']:.4f}")
        print(f"Precision (Weighted): {results['precision_weighted']:.4f}")
        print(f"Recall (Macro): {results['recall_macro']:.4f}")
        print(f"Recall (Weighted): {results['recall_weighted']:.4f}")
        print(f"Mean Prediction Confidence: {results['mean_confidence']:.4f}")
        
        print("\n" + "-"*60)
        print("PER-CLASS PERFORMANCE")
        print("-"*60)
        
        target_names = results['class_names']
        class_report = results['classification_report']
        
        for class_name in target_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"{class_name:15s} - Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}, "
                      f"F1: {metrics['f1-score']:.3f}, "
                      f"Support: {metrics['support']}")
        
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        
        cm = results['confusion_matrix']
        print("   ", "  ".join(f"{name:8s}" for name in target_names))
        for i, row in enumerate(cm):
            print(f"{target_names[i]:8s}", "  ".join(f"{val:8d}" for val in row))
        
        print("="*60)
    
    def analyze_misclassifications(self, results: Dict[str, Any], 
                                 max_examples: int = 5) -> None:
        """
        Analyze and display misclassified examples.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            max_examples (int): Maximum number of examples to show per error type
        """
        if 'test_texts' not in results:
            print("Cannot analyze misclassifications: test texts not available")
            return
        
        test_texts = results['test_texts']
        true_labels = results['true_labels']
        predicted_labels = results['predicted_labels']
        
        print("\n" + "="*60)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*60)
        
        # Find misclassified examples
        misclassified = []
        for i, (true_label, pred_label) in enumerate(zip(true_labels, predicted_labels)):
            if true_label != pred_label:
                misclassified.append({
                    'index': i,
                    'text': test_texts[i],
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': np.max(results['probabilities'][i])
                })
        
        if not misclassified:
            print("No misclassifications found!")
            return
        
        print(f"Total misclassifications: {len(misclassified)}")
        print(f"Showing up to {max_examples} examples:\n")
        
        for i, example in enumerate(misclassified[:max_examples]):
            print(f"Example {i+1}:")
            print(f"  True Label: {example['true_label']}")
            print(f"  Predicted: {example['predicted_label']}")
            print(f"  Confidence: {example['confidence']:.3f}")
            print(f"  Text: {example['text'][:200]}...")
            print()
    
    def save_evaluation_report(self, results: Dict[str, Any], 
                             output_path: str = "evaluation_report.txt") -> None:
        """
        Save evaluation results to a text file.
        
        Args:
            results (Dict[str, Any]): Evaluation results
            output_path (str): Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"F1 Score (Macro): {results['f1_macro']:.4f}\n")
            f.write(f"F1 Score (Weighted): {results['f1_weighted']:.4f}\n")
            f.write(f"Precision (Macro): {results['precision_macro']:.4f}\n")
            f.write(f"Precision (Weighted): {results['precision_weighted']:.4f}\n")
            f.write(f"Recall (Macro): {results['recall_macro']:.4f}\n")
            f.write(f"Recall (Weighted): {results['recall_weighted']:.4f}\n")
            f.write(f"Mean Prediction Confidence: {results['mean_confidence']:.4f}\n\n")
            
            # Detailed classification report
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-"*60 + "\n")
            target_names = results['class_names']
            class_report = results['classification_report']
            
            for class_name in target_names:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    f.write(f"{class_name}: Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, "
                           f"F1={metrics['f1-score']:.3f}, "
                           f"Support={metrics['support']}\n")
        
        print(f"Evaluation report saved to {output_path}")


def main():
    """Main evaluation function."""
    # Configuration
    DATA_PATH = "data/sample_abstracts.csv"
    MODELS_DIR = "models"
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(MODELS_DIR)
        
        # Load models
        evaluator.load_models()
        
        # Evaluate on test data
        results = evaluator.evaluate_on_test_data(DATA_PATH)
        
        # Print comprehensive report
        evaluator.print_evaluation_report(results)
        
        # Analyze misclassifications
        evaluator.analyze_misclassifications(results)
        
        # Save report to file
        evaluator.save_evaluation_report(results)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
