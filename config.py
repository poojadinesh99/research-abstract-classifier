"""
Configuration settings for the research abstract classifier.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Data settings
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "sample_abstracts.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model settings
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.95
NGRAM_RANGE = (1, 2)

# Training settings
MODEL_TYPE = "LogisticRegression"
MAX_ITER = 1000
SOLVER = "liblinear"
C = 1.0
CV_FOLDS = 5

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
CONFIDENCE_THRESHOLD = 0.7

# File names
CLASSIFIER_FILE = "classifier.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
