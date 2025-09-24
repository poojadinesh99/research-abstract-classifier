"""
Training module for research abstract classification.

This module handles model training, validation, and persistence for the
research abstract classifier using scikit-learn.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Tuple, Dict, Any

try:
    from .preprocessing import TextPreprocessor, load_and_preprocess_data
except ImportError:
    from preprocessing import TextPreprocessor, load_and_preprocess_data


class AbstractClassifierTrainer:
    """
    A comprehensive trainer class for research abstract classification.
    
    This class handles the complete training pipeline including data preparation,
    model training, validation, and model persistence.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the classifier trainer.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_data(self, data_path: str, test_size: float = 0.2) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        Load and prepare training data.
        
        Args:
            data_path (str): Path to the CSV data file
            test_size (float): Fraction of data to use for testing
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test, train_texts, test_texts
        """
        print("Loading and preparing data...")
        
        # Load raw data
        texts, labels = load_and_preprocess_data(data_path)
        
        # Adjust test_size for small datasets to ensure minimum samples per class
        unique_labels = set(labels)
        min_samples_per_class = min([labels.count(label) for label in unique_labels])
        
        # Ensure at least 1 sample per class in test set, but not more than half
        adjusted_test_size = min(test_size, 0.5)
        min_test_samples = len(unique_labels)  # At least 1 per class
        
        if len(texts) * adjusted_test_size < min_test_samples:
            adjusted_test_size = min_test_samples / len(texts)
            print(f"   Adjusted test_size to {adjusted_test_size:.2f} for small dataset")
        
        # For very small datasets, use simple split without stratification
        if min_samples_per_class < 2:
            print("   Using simple split (no stratification) for very small dataset")
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels,
                test_size=adjusted_test_size,
                random_state=self.random_state
            )
        else:
            # Split into train/test with stratification
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels,
                test_size=adjusted_test_size,
                random_state=self.random_state,
                stratify=labels
            )
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Testing samples: {len(test_texts)}")
        
        # Encode labels
        y_train = self.label_encoder.fit_transform(train_labels)
        y_test = self.label_encoder.transform(test_labels)
        
        # Fit preprocessor and transform texts
        X_train = self.preprocessor.fit_transform(train_texts)
        X_test = self.preprocessor.transform_texts(test_texts)
        
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test, train_texts, test_texts
    
    def train_model(self, X_train: Any, y_train: Any) -> Dict[str, float]:
        """
        Train the classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dict[str, float]: Cross-validation scores
        """
        print("Training the classifier...")
        
        # Adjust CV folds for small datasets
        n_samples = X_train.shape[0]
        cv_folds = min(5, n_samples // 2)  # Use fewer folds for small datasets
        cv_folds = max(2, cv_folds)  # At least 2 folds
        
        print(f"Using {cv_folds}-fold cross-validation for dataset with {n_samples} training samples")
        
        # Perform cross-validation before final training
        try:
            cv_scores = cross_val_score(
                self.classifier, X_train, y_train,
                cv=cv_folds, scoring='f1_macro'
            )
        except ValueError as e:
            # If stratified CV fails, use simple CV
            print(f"Stratified CV failed ({e}), using simple CV")
            from sklearn.model_selection import KFold
            cv_scores = cross_val_score(
                self.classifier, X_train, y_train,
                cv=KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring='f1_macro'
            )
        
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model on full training set
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        print("Model training completed!")
        
        return {
            'cv_mean_f1': cv_scores.mean(),
            'cv_std_f1': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def evaluate_model(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        target_names = self.label_encoder.classes_
        class_report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True
        )
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1 (macro): {f1_macro:.4f}")
        print(f"Test F1 (weighted): {f1_weighted:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report
        }
    
    def save_models(self, models_dir: str = "models") -> None:
        """
        Save trained models and preprocessor to disk.
        
        Args:
            models_dir (str): Directory to save models
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Save classifier
        classifier_path = os.path.join(models_dir, "classifier.pkl")
        joblib.dump(self.classifier, classifier_path)
        print(f"Classifier saved to {classifier_path}")
        
        # Save label encoder
        encoder_path = os.path.join(models_dir, "label_encoder.pkl")
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Label encoder saved to {encoder_path}")
        
        # Save vectorizer
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        self.preprocessor.save_vectorizer(vectorizer_path)
        
        print("All models saved successfully!")
    
    def load_models(self, models_dir: str = "models") -> 'AbstractClassifierTrainer':
        """
        Load trained models from disk.
        
        Args:
            models_dir (str): Directory containing saved models
            
        Returns:
            AbstractClassifierTrainer: Self for method chaining
        """
        # Load classifier
        classifier_path = os.path.join(models_dir, "classifier.pkl")
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
        self.classifier = joblib.load(classifier_path)
        
        # Load label encoder
        encoder_path = os.path.join(models_dir, "label_encoder.pkl")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
        self.label_encoder = joblib.load(encoder_path)
        
        # Load vectorizer
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        self.preprocessor.load_vectorizer(vectorizer_path)
        
        self.is_trained = True
        print("All models loaded successfully!")
        
        return self


def main():
    """Main training function."""
    # Configuration
    DATA_PATH = "data/sample_abstracts.csv"
    MODELS_DIR = "models"
    
    try:
        # Initialize trainer
        trainer = AbstractClassifierTrainer()
        
        # Prepare data
        X_train, X_test, y_train, y_test, train_texts, test_texts = trainer.prepare_data(DATA_PATH)
        
        # Train model
        cv_results = trainer.train_model(X_train, y_train)
        
        # Evaluate model
        test_results = trainer.evaluate_model(X_test, y_test)
        
        # Save models
        trainer.save_models(MODELS_DIR)
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Cross-validation F1 (macro): {cv_results['cv_mean_f1']:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")
        print(f"Models saved to: {MODELS_DIR}/")
        print("="*50)
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
