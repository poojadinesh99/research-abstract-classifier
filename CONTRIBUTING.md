# Contributing to Research Abstract Classifier

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Docker (optional, for containerized development)

### Quick Start
```bash
git clone https://github.com/poojadinesh99/research-abstract-classifier.git
cd research-abstract-classifier
make setup  # Creates venv and installs dependencies
make train  # Train the model
make test   # Run test suite
make api    # Start development server
```

## Development Workflow

### 1. Code Style
- Follow PEP 8 guidelines
- Use type hints where applicable
- Maximum line length: 127 characters
- Use descriptive variable and function names

### 2. Testing
- Write unit tests for all new functions
- Maintain >80% code coverage
- Test both happy path and edge cases
- Run `make test` before submitting PRs

### 3. Documentation
- Update README.md for new features
- Add docstrings to all public functions
- Update API documentation for endpoint changes
- Include examples in docstrings

## Project Structure

```
research-abstract-classifier/
├── src/                    # Core ML pipeline
│   ├── preprocessing.py    # Text preprocessing utilities
│   ├── train.py           # Model training pipeline
│   ├── evaluate.py        # Model evaluation tools
│   └── inference.py       # Prediction utilities
├── api/                   # FastAPI application
│   └── app.py            # REST API endpoints
├── tests/                 # Test suite
│   └── test_pipeline.py  # Comprehensive tests
├── data/                  # Sample datasets
├── models/               # Trained model artifacts
├── .github/workflows/    # CI/CD configuration
└── docs/                # Additional documentation
```

## Adding New Features

### 1. New ML Models
- Add model classes to `src/models/`
- Update training pipeline in `src/train.py`
- Add evaluation metrics in `src/evaluate.py`
- Update API endpoints if needed

### 2. New API Endpoints
- Add routes to `api/app.py`
- Include proper error handling
- Add Pydantic models for request/response
- Write integration tests

### 3. New Data Sources
- Add preprocessing utilities to `src/preprocessing.py`
- Ensure consistent data format
- Update documentation

## Code Review Guidelines

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally and in CI
- [ ] Documentation updated
- [ ] No sensitive data committed
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

### Review Focus Areas
- Code quality and maintainability
- Test coverage and quality
- Performance implications
- Security considerations
- Documentation completeness

## Release Process

1. **Feature Development**: Work on feature branches
2. **Testing**: Ensure all tests pass
3. **Code Review**: Submit PR for review
4. **Integration**: Merge to main branch
5. **Deployment**: Automatic CI/CD pipeline deployment
6. **Monitoring**: Monitor performance metrics

## Getting Help

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Check existing documentation first
- Provide minimal reproducible examples for bugs
