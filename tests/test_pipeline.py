"""
Comprehensive test suite for the research abstract classification pipeline.

This module contains end-to-end tests to ensure the complete pipeline works
correctly from training to inference.
"""

import os
import sys
import tempfile
import shutil
import pytest
import pandas as pd
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from preprocessing import TextPreprocessor, load_and_preprocess_data
    from train import AbstractClassifierTrainer
    from evaluate import ModelEvaluator
    from inference import AbstractClassifier
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all dependencies are installed and src modules are accessible.")
    sys.exit(1)


class TestPipeline:
    """Test class for the complete ML pipeline."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        # Create temporary directory for test models
        cls.temp_dir = tempfile.mkdtemp()
        cls.models_dir = os.path.join(cls.temp_dir, "test_models")
        
        # Create sample test data
        cls.sample_data = [
            ("Deep learning models have shown remarkable performance in natural language processing tasks.", "Computer Science"),
            ("The human brain exhibits complex neural networks that process information through interconnected neurons.", "Biology"),
            ("Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level.", "Physics"),
            ("Graph theory provides mathematical frameworks for analyzing network structures and relationships.", "Mathematics"),
            ("Organic synthesis of complex molecular structures requires precise control of reaction conditions.", "Chemistry"),
            ("Cardiovascular disease remains a leading cause of mortality worldwide.", "Medicine"),
            ("Machine learning algorithms can identify patterns in large datasets through statistical analysis.", "Computer Science"),
            ("Cellular metabolism involves complex biochemical pathways that regulate energy production.", "Biology"),
        ]
        
        # Create temporary CSV file
        cls.test_data_path = os.path.join(cls.temp_dir, "test_data.csv")
        df = pd.DataFrame(cls.sample_data, columns=['abstract', 'category'])
        df.to_csv(cls.test_data_path, index=False)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test fixtures."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
    
    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        # Initialize preprocessor
        preprocessor = TextPreprocessor(max_features=100)
        
        # Test text cleaning
        dirty_text = "This is a TEST with 123 numbers and https://example.com URLs!"
        cleaned = preprocessor.clean_text(dirty_text)
        
        assert "test" in cleaned.lower()
        assert "123" not in cleaned
        assert "https://example.com" not in cleaned
        
        # Test tokenization and stopword removal
        tokens = preprocessor.tokenize_and_remove_stopwords("This is a test document with some stopwords.")
        
        assert "test" in tokens
        assert "document" in tokens
        assert "is" not in tokens  # stopword
        assert "a" not in tokens   # stopword
        
        # Test full preprocessing
        processed = preprocessor.preprocess_text(dirty_text)
        assert len(processed) > 0
        assert isinstance(processed, str)
    
    def test_data_loading(self):
        """Test data loading functionality."""
        texts, labels = load_and_preprocess_data(self.test_data_path)
        
        assert len(texts) == len(self.sample_data)
        assert len(labels) == len(self.sample_data)
        assert isinstance(texts, list)
        assert isinstance(labels, list)
        
        # Check that we have the expected categories
        unique_labels = set(labels)
        expected_categories = {"Computer Science", "Biology", "Physics", "Mathematics", "Chemistry", "Medicine"}
        assert unique_labels.issubset(expected_categories)
    
    def test_vectorizer_fit_transform(self):
        """Test TF-IDF vectorizer functionality."""
        texts, _ = load_and_preprocess_data(self.test_data_path)
        
        preprocessor = TextPreprocessor(max_features=50)
        
        # Test fit_transform
        X = preprocessor.fit_transform(texts)
        
        assert X.shape[0] == len(texts)
        assert X.shape[1] <= 50  # max_features
        assert hasattr(preprocessor, 'vectorizer')
        assert preprocessor.vectorizer is not None
        
        # Test transform on new data
        new_texts = ["This is a test document about machine learning."]
        X_new = preprocessor.transform_texts(new_texts)
        
        assert X_new.shape[0] == 1
        assert X_new.shape[1] == X.shape[1]  # Same number of features
    
    def test_model_training(self):
        """Test model training functionality."""
        trainer = AbstractClassifierTrainer(random_state=42)
        
        # Prepare data
        X_train, X_test, y_train, y_test, train_texts, test_texts = trainer.prepare_data(
            self.test_data_path, test_size=0.25
        )
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Train model
        cv_results = trainer.train_model(X_train, y_train)
        
        assert trainer.is_trained
        assert 'cv_mean_f1' in cv_results
        assert isinstance(cv_results['cv_mean_f1'], float)
        assert 0 <= cv_results['cv_mean_f1'] <= 1
        
        # Evaluate model
        test_results = trainer.evaluate_model(X_test, y_test)
        
        assert 'accuracy' in test_results
        assert 'f1_macro' in test_results
        assert isinstance(test_results['accuracy'], float)
        assert 0 <= test_results['accuracy'] <= 1
        
        # Save models
        trainer.save_models(self.models_dir)
        
        # Check that model files were created
        assert os.path.exists(os.path.join(self.models_dir, "classifier.pkl"))
        assert os.path.exists(os.path.join(self.models_dir, "vectorizer.pkl"))
        assert os.path.exists(os.path.join(self.models_dir, "label_encoder.pkl"))
    
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        # First train a model (using the previous test)
        self.test_model_training()
        
        # Test evaluator
        evaluator = ModelEvaluator(self.models_dir)
        evaluator.load_models()
        
        assert evaluator.is_loaded
        
        # Evaluate on test data
        results = evaluator.evaluate_on_test_data(self.test_data_path, test_size=0.25, random_state=42)
        
        assert 'accuracy' in results
        assert 'f1_macro' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results
        
        # Test custom evaluation
        texts, labels = load_and_preprocess_data(self.test_data_path)
        custom_results = evaluator.evaluate_on_custom_data(texts[:2], labels[:2])
        
        assert 'accuracy' in custom_results
        assert len(custom_results['true_labels']) == 2
    
    def test_inference_pipeline(self):
        """Test inference functionality."""
        # Ensure models are trained
        self.test_model_training()
        
        # Test inference
        classifier = AbstractClassifier(self.models_dir)
        classifier.load_models()
        
        assert classifier.is_loaded
        
        # Test single prediction
        test_text = "This research presents a novel deep learning approach for computer vision tasks."
        result = classifier.predict_single(test_text)
        
        assert 'category' in result
        assert 'confidence' in result
        assert isinstance(result['category'], str)
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        
        # Test batch prediction
        test_texts = [
            "Deep learning models for natural language processing.",
            "Biological systems and neural networks in the brain."
        ]
        batch_results = classifier.predict_batch(test_texts)
        
        assert len(batch_results) == 2
        for result in batch_results:
            assert 'category' in result
            assert 'confidence' in result
        
        # Test detailed prediction
        detailed_result = classifier.predict_with_all_probabilities(test_text)
        
        assert 'category' in detailed_result
        assert 'confidence' in detailed_result
        assert 'all_probabilities' in detailed_result
        assert isinstance(detailed_result['all_probabilities'], dict)
        
        # Test available categories
        categories = classifier.get_available_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        
        # Test confidence checking
        high_conf = classifier.is_high_confidence(result, threshold=0.5)
        assert isinstance(high_conf, bool)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # 1. Load and preprocess data
        texts, labels = load_and_preprocess_data(self.test_data_path)
        assert len(texts) > 0
        
        # 2. Train model
        trainer = AbstractClassifierTrainer(random_state=42)
        X_train, X_test, y_train, y_test, _, _ = trainer.prepare_data(
            self.test_data_path, test_size=0.25
        )
        
        cv_results = trainer.train_model(X_train, y_train)
        test_results = trainer.evaluate_model(X_test, y_test)
        trainer.save_models(self.models_dir)
        
        # 3. Load models and make predictions
        classifier = AbstractClassifier(self.models_dir)
        classifier.load_models()
        
        # 4. Test on new data
        new_abstracts = [
            "Machine learning algorithms for predictive analytics and data mining applications.",
            "Study of protein folding mechanisms in cellular biology and biochemistry.",
            "Quantum entanglement properties in theoretical physics research."
        ]
        
        predictions = classifier.predict_batch(new_abstracts)
        
        # Verify results
        assert len(predictions) == 3
        for pred in predictions:
            assert pred['category'] in classifier.get_available_categories()
            assert 0 <= pred['confidence'] <= 1
        
        # 5. Evaluate loaded model
        evaluator = ModelEvaluator(self.models_dir)
        evaluator.load_models()
        
        eval_results = evaluator.evaluate_on_test_data(self.test_data_path, test_size=0.25, random_state=42)
        
        # Results should be consistent
        assert abs(eval_results['accuracy'] - test_results['accuracy']) < 0.01
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test with non-existent data file
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_data("non_existent_file.csv")
        
        # Test inference without loading models
        classifier = AbstractClassifier("non_existent_dir")
        with pytest.raises(ValueError):
            classifier.predict_single("test text")
        
        # Test with empty text
        self.test_model_training()  # Ensure models exist
        classifier = AbstractClassifier(self.models_dir)
        classifier.load_models()
        
        result = classifier.predict_single("")
        assert 'category' in result  # Should handle empty string gracefully
        
        # Test with very short text
        result = classifier.predict_single("AI")
        assert 'category' in result
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Train and save models
        trainer = AbstractClassifierTrainer(random_state=42)
        X_train, X_test, y_train, y_test, _, _ = trainer.prepare_data(self.test_data_path)
        trainer.train_model(X_train, y_train)
        trainer.save_models(self.models_dir)
        
        # Create new trainer and load models
        new_trainer = AbstractClassifierTrainer()
        new_trainer.load_models(self.models_dir)
        
        assert new_trainer.is_trained
        
        # Test that loaded model makes same predictions
        test_results1 = trainer.evaluate_model(X_test, y_test)
        test_results2 = new_trainer.evaluate_model(X_test, y_test)
        
        # Results should be identical
        assert abs(test_results1['accuracy'] - test_results2['accuracy']) < 1e-10


def test_sample_predictions():
    """Integration test with sample abstracts."""
    # This test can run independently if models exist
    data_path = "data/sample_abstracts.csv"
    models_dir = "models"
    
    # Skip if data or models don't exist
    if not os.path.exists(data_path):
        pytest.skip("Sample data not found")
    
    # Try to use existing models or train new ones
    if not os.path.exists(os.path.join(models_dir, "classifier.pkl")):
        # Train models if they don't exist
        trainer = AbstractClassifierTrainer()
        X_train, X_test, y_train, y_test, _, _ = trainer.prepare_data(data_path)
        trainer.train_model(X_train, y_train)
        trainer.save_models(models_dir)
    
    # Test inference with existing models
    classifier = AbstractClassifier(models_dir)
    classifier.load_models()
    
    # Test sample predictions
    test_abstracts = [
        "This paper presents a novel neural network architecture for image classification using convolutional layers.",
        "We studied the genetic mechanisms underlying cellular differentiation in stem cell biology.",
        "The research investigates quantum field theory applications in particle physics experiments."
    ]
    
    predictions = classifier.predict_batch(test_abstracts)
    
    # Verify reasonable predictions
    assert len(predictions) == 3
    
    # First should likely be Computer Science
    assert predictions[0]['category'] in ['Computer Science', 'Mathematics']
    
    # Second should likely be Biology or Medicine
    assert predictions[1]['category'] in ['Biology', 'Medicine']
    
    # Third should likely be Physics
    assert predictions[2]['category'] in ['Physics', 'Mathematics']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
