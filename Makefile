# Research Abstract Classifier - Makefile
# Common operations for the ML pipeline

.PHONY: help install install-dev setup train evaluate test api clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  help         - Show this help message"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Set up development environment"
	@echo "  train        - Train the classification model"
	@echo "  evaluate     - Evaluate model performance"
	@echo "  test         - Run test suite"
	@echo "  api          - Start the FastAPI server"
	@echo "  clean        - Clean temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy matplotlib seaborn
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

setup: install-dev
	@echo "Setting up development environment..."
	@echo "Creating models directory..."
	mkdir -p models
	@echo "Development environment ready!"

train:
	@echo "Training classification model..."
	python src/train.py

evaluate:
	@echo "Evaluating model performance..."
	python src/evaluate.py

test:
	@echo "Running test suite..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

api:
	@echo "Starting FastAPI server..."
	python api/app.py

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -f evaluation_report.txt

docker-build:
	@echo "Building Docker image..."
	docker build -t research-abstract-classifier .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 research-abstract-classifier

docker-compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up --build

docker-compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

# Linting and formatting
lint:
	@echo "Running linters..."
	flake8 src/ api/ tests/
	mypy src/ api/

format:
	@echo "Formatting code..."
	black src/ api/ tests/

# Quick pipeline test
quick-test:
	@echo "Running quick pipeline test..."
	python -c "from src.preprocessing import TextPreprocessor; print('✓ Preprocessing module works')"
	python -c "from src.train import AbstractClassifierTrainer; print('✓ Training module works')"
	python -c "from src.evaluate import ModelEvaluator; print('✓ Evaluation module works')"
	python -c "from src.inference import AbstractClassifier; print('✓ Inference module works')"

# Full pipeline execution
full-pipeline: clean train evaluate test
	@echo "Full pipeline completed successfully!"

# Development workflow
dev-setup: setup quick-test
	@echo "Development setup completed!"

# Production deployment
deploy: clean install train
	@echo "Production deployment ready!"
