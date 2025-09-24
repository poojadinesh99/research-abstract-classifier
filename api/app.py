"""
FastAPI application for research abstract classification.

This module provides a REST API for classifying research abstracts using
the trained machine learning model.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from inference import AbstractClassifier
except ImportError as e:
    print(f"Error importing inference module: {e}")
    print("Make sure the src directory is accessible and dependencies are installed.")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Research Abstract Classifier API",
    description="A machine learning API for classifying research abstracts into academic categories",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = None


class PredictionRequest(BaseModel):
    """Request model for single abstract prediction."""
    abstract: str = Field(..., description="Research abstract text to classify", min_length=10)


class BatchPredictionRequest(BaseModel):
    """Request model for batch abstract prediction."""
    abstracts: List[str] = Field(..., description="List of research abstracts to classify", min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    high_confidence: bool = Field(..., description="Whether prediction has high confidence (>0.7)")


class DetailedPredictionResponse(BaseModel):
    """Response model for detailed prediction with all probabilities."""
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    high_confidence: bool = Field(..., description="Whether prediction has high confidence (>0.7)")
    all_probabilities: Dict[str, float] = Field(..., description="Probabilities for all categories")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total number of abstracts processed")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Status message")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    available_categories: Optional[List[str]] = Field(None, description="Available classification categories")


@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup."""
    global classifier
    
    try:
        logger.info("Loading machine learning models...")
        
        # Determine models directory path
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Initialize and load classifier
        classifier = AbstractClassifier(models_dir)
        classifier.load_models()
        
        logger.info("Models loaded successfully!")
        logger.info(f"Available categories: {classifier.get_available_categories()}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Don't exit - let health check show the error
        classifier = None


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and model status.
    
    Returns:
        HealthResponse: API and model status information
    """
    global classifier
    
    if classifier is None or not classifier.is_loaded:
        return HealthResponse(
            status="unhealthy",
            message="Machine learning model not loaded",
            model_loaded=False,
            available_categories=None
        )
    
    try:
        categories = classifier.get_available_categories()
        return HealthResponse(
            status="healthy",
            message="API is running and model is loaded",
            model_loaded=True,
            available_categories=categories
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            model_loaded=False,
            available_categories=None
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_abstract(request: PredictionRequest):
    """
    Classify a single research abstract.
    
    Args:
        request (PredictionRequest): Request containing abstract text
        
    Returns:
        PredictionResponse: Prediction result with confidence score
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    global classifier
    
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make prediction
        result = classifier.predict_single(request.abstract)
        
        # Check confidence
        high_confidence = classifier.is_high_confidence(result)
        
        logger.info(f"Predicted category: {result['category']} (confidence: {result['confidence']:.3f})")
        
        return PredictionResponse(
            category=result['category'],
            confidence=result['confidence'],
            high_confidence=high_confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/detailed", response_model=DetailedPredictionResponse)
async def predict_abstract_detailed(request: PredictionRequest):
    """
    Classify a research abstract with detailed probability breakdown.
    
    Args:
        request (PredictionRequest): Request containing abstract text
        
    Returns:
        DetailedPredictionResponse: Detailed prediction with all class probabilities
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    global classifier
    
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make detailed prediction
        result = classifier.predict_with_all_probabilities(request.abstract)
        
        # Check confidence
        high_confidence = classifier.is_high_confidence(result)
        
        logger.info(f"Detailed prediction - Category: {result['category']} (confidence: {result['confidence']:.3f})")
        
        return DetailedPredictionResponse(
            category=result['category'],
            confidence=result['confidence'],
            high_confidence=high_confidence,
            all_probabilities=result['all_probabilities']
        )
        
    except Exception as e:
        logger.error(f"Detailed prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_abstracts_batch(request: BatchPredictionRequest):
    """
    Classify multiple research abstracts in batch.
    
    Args:
        request (BatchPredictionRequest): Request containing list of abstracts
        
    Returns:
        BatchPredictionResponse: Batch prediction results
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    global classifier
    
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Make batch predictions
        results = classifier.predict_batch(request.abstracts)
        
        # Convert to response format
        predictions = []
        for result in results:
            high_confidence = classifier.is_high_confidence(result)
            predictions.append(PredictionResponse(
                category=result['category'],
                confidence=result['confidence'],
                high_confidence=high_confidence
            ))
        
        logger.info(f"Processed batch of {len(request.abstracts)} abstracts")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(request.abstracts)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/categories", response_model=List[str])
async def get_categories():
    """
    Get list of available classification categories.
    
    Returns:
        List[str]: List of available category names
        
    Raises:
        HTTPException: If model not loaded
    """
    global classifier
    
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        categories = classifier.get_available_categories()
        return categories
        
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        Dict: Model information and statistics
    """
    global classifier
    
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = {
            "model_type": "Logistic Regression",
            "preprocessing": "TF-IDF Vectorization",
            "categories": classifier.get_available_categories(),
            "num_categories": len(classifier.get_available_categories()),
            "status": "loaded"
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


def main():
    """Run the FastAPI application."""
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting Research Abstract Classifier API on {host}:{port}")
    print("API Documentation available at: http://localhost:8000/docs")
    
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )


if __name__ == "__main__":
    main()
