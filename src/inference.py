"""
Inference module for research abstract classification.

This module provides functionality to load trained models and make predictions
on new research abstracts.
"""

import os
import joblib
import argparse
import numpy as np
from typing import Dict, List, Tuple, Union

try:
    from .preprocessing import TextPreprocessor
except ImportError:
    from preprocessing import TextPreprocessor


class AbstractClassifier:
    """
    Research abstract classifier for inference on new texts.
    
    This class loads pre-trained models and provides easy-to-use prediction
    functionality with confidence scores.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the classifier.
        
        Args:
            models_dir (str): Directory containing saved models
        """
        self.models_dir = models_dir
        self.classifier = None
        self.label_encoder = None
        self.preprocessor = TextPreprocessor()
        self.is_loaded = False
        
    def load_models(self) -> 'AbstractClassifier':
        """
        Load all trained models from disk.
        
        Returns:
            AbstractClassifier: Self for method chaining
        """
        print("Loading trained models...")
        
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
        print(f"Models loaded successfully!")
        print(f"Available categories: {list(self.label_encoder.classes_)}")
        
        return self
    
    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predict category for a single research abstract.
        
        Args:
            text (str): Research abstract text
            
        Returns:
            Dict[str, Union[str, float]]: Prediction with confidence
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Preprocess and transform text
        X = self.preprocessor.transform_texts([text])
        
        # Make prediction
        y_pred = self.classifier.predict(X)[0]
        y_pred_proba = self.classifier.predict_proba(X)[0]
        
        # Get predicted category and confidence
        predicted_category = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = float(np.max(y_pred_proba))
        
        return {
            'category': predicted_category,
            'confidence': confidence
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Predict categories for multiple research abstracts.
        
        Args:
            texts (List[str]): List of research abstract texts
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of predictions with confidence
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        if not texts:
            return []
        
        # Preprocess and transform texts
        X = self.preprocessor.transform_texts(texts)
        
        # Make predictions
        y_pred = self.classifier.predict(X)
        y_pred_proba = self.classifier.predict_proba(X)
        
        # Convert predictions to readable format
        results = []
        for i, (pred, proba) in enumerate(zip(y_pred, y_pred_proba)):
            predicted_category = self.label_encoder.inverse_transform([pred])[0]
            confidence = float(np.max(proba))
            
            results.append({
                'category': predicted_category,
                'confidence': confidence
            })
        
        return results
    
    def predict_with_all_probabilities(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict category with probabilities for all classes.
        
        Args:
            text (str): Research abstract text
            
        Returns:
            Dict: Prediction with all class probabilities
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Preprocess and transform text
        X = self.preprocessor.transform_texts([text])
        
        # Make prediction
        y_pred = self.classifier.predict(X)[0]
        y_pred_proba = self.classifier.predict_proba(X)[0]
        
        # Get predicted category and confidence
        predicted_category = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = float(np.max(y_pred_proba))
        
        # Get probabilities for all classes
        class_probabilities = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_probabilities[class_name] = float(y_pred_proba[i])
        
        return {
            'category': predicted_category,
            'confidence': confidence,
            'all_probabilities': class_probabilities
        }
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available prediction categories.
        
        Returns:
            List[str]: List of category names
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        return list(self.label_encoder.classes_)
    
    def is_high_confidence(self, prediction: Dict[str, Union[str, float]], 
                          threshold: float = 0.7) -> bool:
        """
        Check if prediction has high confidence.
        
        Args:
            prediction (Dict): Prediction result from predict_single
            threshold (float): Confidence threshold
            
        Returns:
            bool: True if confidence is above threshold
        """
        return prediction['confidence'] >= threshold


def main():
    """Main inference function with command-line interface."""
    parser = argparse.ArgumentParser(description='Research Abstract Classifier - Inference')
    parser.add_argument('--text', type=str, help='Abstract text to classify')
    parser.add_argument('--file', type=str, help='File containing abstract text')
    parser.add_argument('--models-dir', type=str, default='models', 
                       help='Directory containing trained models')
    parser.add_argument('--show-all-probs', action='store_true',
                       help='Show probabilities for all categories')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        print("Please provide either --text or --file argument")
        return
    
    try:
        # Initialize classifier
        classifier = AbstractClassifier(args.models_dir)
        classifier.load_models()
        
        # Get text to classify
        if args.file:
            if not os.path.exists(args.file):
                print(f"File not found: {args.file}")
                return
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            text = args.text
        
        if not text:
            print("Empty text provided")
            return
        
        # Make prediction
        if args.show_all_probs:
            result = classifier.predict_with_all_probabilities(text)
            
            print("\n" + "="*50)
            print("CLASSIFICATION RESULT")
            print("="*50)
            print(f"Predicted Category: {result['category']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"High Confidence: {classifier.is_high_confidence(result)}")
            print("\nAll Category Probabilities:")
            print("-" * 30)
            
            # Sort probabilities by value
            sorted_probs = sorted(result['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            
            for category, prob in sorted_probs:
                print(f"{category:20s}: {prob:.4f}")
            
        else:
            result = classifier.predict_single(text)
            
            print("\n" + "="*50)
            print("CLASSIFICATION RESULT")
            print("="*50)
            print(f"Predicted Category: {result['category']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"High Confidence: {classifier.is_high_confidence(result)}")
        
        print("="*50)
        
        # Show input text (truncated)
        print(f"\nInput Text (first 200 chars):")
        print(f"'{text[:200]}{'...' if len(text) > 200 else ''}'")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    # If no command line arguments, run interactive mode
    import sys
    
    if len(sys.argv) == 1:
        print("Research Abstract Classifier - Interactive Mode")
        print("=" * 50)
        
        try:
            # Initialize classifier
            classifier = AbstractClassifier()
            classifier.load_models()
            
            print(f"Available categories: {', '.join(classifier.get_available_categories())}")
            print("\nEnter research abstracts to classify (type 'quit' to exit):")
            
            while True:
                text = input("\n> Abstract: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                try:
                    result = classifier.predict_single(text)
                    print(f"  Predicted: {result['category']} (confidence: {result['confidence']:.3f})")
                    
                    if not classifier.is_high_confidence(result):
                        print("  ⚠️  Low confidence prediction")
                
                except Exception as e:
                    print(f"  Error: {e}")
            
            print("Goodbye!")
            
        except Exception as e:
            print(f"Error initializing classifier: {e}")
    else:
        main()
