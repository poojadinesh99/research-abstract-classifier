"""
Advanced model configuration and hyperparameter management.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import os

@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters."""
    
    # TF-IDF Parameters
    max_features: int = 10000
    ngram_range: tuple = (1, 2)
    min_df: int = 2
    max_df: float = 0.8
    
    # Logistic Regression Parameters
    C: float = 1.0
    max_iter: int = 1000
    solver: str = 'lbfgs'
    multi_class: str = 'ovr'
    random_state: int = 42
    
    # Training Parameters
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = 'f1_macro'
    
    # Performance Thresholds
    min_accuracy: float = 0.75
    min_f1_score: float = 0.70
    confidence_threshold: float = 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

@dataclass
class ExperimentResults:
    """Track experiment results and metrics."""
    
    experiment_id: str
    config: Dict[str, Any]
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    training_time: float
    model_size_mb: float
    feature_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Default configurations for different scenarios
DEFAULT_CONFIG = ModelConfig()

FAST_CONFIG = ModelConfig(
    max_features=5000,
    ngram_range=(1, 1),
    C=0.1,
    max_iter=500
)

HIGH_ACCURACY_CONFIG = ModelConfig(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=1,
    C=10.0,
    max_iter=2000,
    cv_folds=10
)
