"""
Text preprocessing module for research abstract classification.

This module provides functions for cleaning and vectorizing text data,
including tokenization, stopword removal, and TF-IDF vectorization.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import List, Tuple
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for research abstracts.
    
    This class handles text cleaning, tokenization, and TF-IDF vectorization
    for machine learning pipelines.
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.95):
        """
        Initialize the text preprocessor.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            min_df (int): Minimum document frequency for TF-IDF
            max_df (float): Maximum document frequency for TF-IDF
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers but keep words with numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove punctuation except hyphens in words
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text: str) -> List[str]:
        """
        Tokenize text and remove stopwords.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            List[str]: List of cleaned tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [
            token for token in tokens 
            if token not in self.stop_words 
            and len(token) > 2 
            and token.isalpha()
        ]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline for a single text.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and remove stopwords
        tokens = self.tokenize_and_remove_stopwords(cleaned_text)
        
        # Join tokens back into string
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def fit_vectorizer(self, texts: List[str]) -> 'TextPreprocessor':
        """
        Fit the TF-IDF vectorizer on training texts.
        
        Args:
            texts (List[str]): List of training texts
            
        Returns:
            TextPreprocessor: Self for method chaining
        """
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english'
        )
        
        self.vectorizer.fit(processed_texts)
        
        return self
    
    def transform_texts(self, texts: List[str]):
        """
        Transform texts using the fitted vectorizer.
        
        Args:
            texts (List[str]): List of texts to transform
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer() first.")
            
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform using fitted vectorizer
        features = self.vectorizer.transform(processed_texts)
        
        return features
    
    def fit_transform(self, texts: List[str]):
        """
        Fit vectorizer and transform texts in one step.
        
        Args:
            texts (List[str]): List of texts to fit and transform
            
        Returns:
            scipy.sparse matrix: TF-IDF features
        """
        self.fit_vectorizer(texts)
        return self.transform_texts(texts)
    
    def save_vectorizer(self, filepath: str) -> None:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("No vectorizer to save. Fit the vectorizer first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str) -> 'TextPreprocessor':
        """
        Load a fitted vectorizer from disk.
        
        Args:
            filepath (str): Path to load the vectorizer from
            
        Returns:
            TextPreprocessor: Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vectorizer file not found: {filepath}")
            
        self.vectorizer = joblib.load(filepath)
        print(f"Vectorizer loaded from {filepath}")
        
        return self


def load_and_preprocess_data(data_path: str) -> Tuple[List[str], List[str]]:
    """
    Load and preprocess data from CSV file.
    
    Args:
        data_path (str): Path to the CSV data file
        
    Returns:
        Tuple[List[str], List[str]]: Texts and labels
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Validate required columns
    if 'abstract' not in df.columns or 'category' not in df.columns:
        raise ValueError("CSV file must contain 'abstract' and 'category' columns")
    
    # Extract texts and labels
    texts = df['abstract'].tolist()
    labels = df['category'].tolist()
    
    print(f"Loaded {len(texts)} samples from {data_path}")
    print(f"Categories: {sorted(set(labels))}")
    
    return texts, labels


if __name__ == "__main__":
    # Example usage
    data_path = "../data/sample_abstracts.csv"
    
    try:
        # Load data
        texts, labels = load_and_preprocess_data(data_path)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Fit and transform texts
        X = preprocessor.fit_transform(texts)
        
        print(f"Feature matrix shape: {X.shape}")
        print("Preprocessing completed successfully!")
        
    except FileNotFoundError:
        print(f"Data file not found. Please ensure {data_path} exists.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
