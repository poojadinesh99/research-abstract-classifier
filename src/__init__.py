"""
Research Abstract Classifier Package

A machine learning pipeline for classifying research abstracts into academic categories.
"""

__version__ = "1.0.0"
__author__ = "Research Abstract Classifier Team"

from .preprocessing import TextPreprocessor, load_and_preprocess_data
from .train import AbstractClassifierTrainer
from .evaluate import ModelEvaluator
from .inference import AbstractClassifier

__all__ = [
    'TextPreprocessor',
    'load_and_preprocess_data',
    'AbstractClassifierTrainer',
    'ModelEvaluator',
    'AbstractClassifier'
]
