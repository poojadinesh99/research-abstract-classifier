#!/usr/bin/env python3
"""
Demo script for the Research Abstract Classifier.

This script demonstrates the complete pipeline from training to inference.
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the complete demo pipeline."""
    
    print("="*60)
    print("RESEARCH ABSTRACT CLASSIFIER - DEMO")
    print("="*60)
    
    try:
        # Check if dependencies are available
        print("\n1. Checking dependencies...")
        
        import pandas as pd
        import sklearn
        import nltk
        print("   ✓ All core dependencies available")
        
        # Import our modules
        from preprocessing import TextPreprocessor, load_and_preprocess_data
        from train import AbstractClassifierTrainer
        from inference import AbstractClassifier
        print("   ✓ All custom modules imported successfully")
        
    except ImportError as e:
        print(f"   ❌ Missing dependencies: {e}")
        print("\nPlease install dependencies with:")
        print("   pip install -r requirements.txt")
        return
    
    # Configuration
    DATA_PATH = "data/sample_abstracts.csv"
    MODELS_DIR = "models"
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ Data file not found: {DATA_PATH}")
        print("Please ensure the sample data file exists.")
        return
    
    print(f"\n2. Loading data from {DATA_PATH}...")
    try:
        texts, labels = load_and_preprocess_data(DATA_PATH)
        print(f"   ✓ Loaded {len(texts)} abstracts with {len(set(labels))} categories")
        print(f"   Categories: {', '.join(sorted(set(labels)))}")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return
    
    print("\n3. Training the model...")
    try:
        # Initialize trainer
        trainer = AbstractClassifierTrainer()
        
        # Prepare data
        X_train, X_test, y_train, y_test, train_texts, test_texts = trainer.prepare_data(DATA_PATH)
        
        # Train model
        cv_results = trainer.train_model(X_train, y_train)
        print(f"   ✓ Model trained with CV F1 score: {cv_results['cv_mean_f1']:.3f}")
        
        # Evaluate
        test_results = trainer.evaluate_model(X_test, y_test)
        print(f"   ✓ Test accuracy: {test_results['accuracy']:.3f}")
        
        # Save models
        trainer.save_models(MODELS_DIR)
        print(f"   ✓ Models saved to {MODELS_DIR}/")
        
    except Exception as e:
        print(f"   ❌ Error during training: {e}")
        return
    
    print("\n4. Testing inference...")
    try:
        # Load classifier
        classifier = AbstractClassifier(MODELS_DIR)
        classifier.load_models()
        print(f"   ✓ Models loaded successfully")
        
        # Test predictions
        test_abstracts = [
            "This paper presents a novel deep learning approach for natural language processing tasks using transformer architectures.",
            "We investigated the cellular mechanisms of protein folding in biological systems using advanced microscopy techniques.",
            "The research explores quantum entanglement properties in multi-particle systems for quantum computing applications.",
            "This study analyzes mathematical models for optimization problems in graph theory and network analysis.",
            "We synthesized novel organic compounds with improved catalytic properties for industrial chemical processes.",
            "The clinical trial evaluated the effectiveness of a new drug therapy for cardiovascular disease treatment."
        ]
        
        print(f"\n   Testing predictions on {len(test_abstracts)} sample abstracts:")
        
        for i, abstract in enumerate(test_abstracts):
            result = classifier.predict_single(abstract)
            confidence_indicator = "🟢" if result['confidence'] > 0.7 else "🟡" if result['confidence'] > 0.5 else "🔴"
            
            print(f"\n   Abstract {i+1}:")
            print(f"   Text: {abstract[:80]}...")
            print(f"   Prediction: {result['category']} {confidence_indicator}")
            print(f"   Confidence: {result['confidence']:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error during inference: {e}")
        return
    
    print("\n5. API Demo...")
    try:
        print("   You can start the API server with:")
        print(f"   python api/app.py")
        print("\n   Then test it with:")
        print('   curl -X POST "http://localhost:8000/predict" \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"abstract": "Your research abstract here"}\'')
        
    except Exception as e:
        print(f"   ❌ Error checking API: {e}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    
    print("\nNext steps:")
    print("• Train with your own data by replacing data/sample_abstracts.csv")
    print("• Start the API server: python api/app.py")
    print("• Run tests: pytest tests/")
    print("• Check API docs: http://localhost:8000/docs")
    print("• Use Docker: docker build -t abstract-classifier .")
    

if __name__ == "__main__":
    main()
