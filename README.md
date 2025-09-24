# Research Abstract Classifier

[![CI/CD Pipeline](https://github.com/poojadinesh99/research-abstract-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/poojadinesh99/research-abstract-classifier/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)

A production-ready machine learning pipeline for classifying research abstracts into academic disciplines using Natural Language Processing, scikit-learn, and FastAPI. This project demonstrates end-to-end ML engineering practices including data preprocessing, model training, RESTful API development, containerization, and automated testing.

## 🏗️ Architecture & Technical Highlights

### **Core ML Pipeline**
- **Advanced NLP Preprocessing**: Custom text cleaning, tokenization, stopword removal, and TF-IDF vectorization
- **Scikit-learn Integration**: Logistic Regression with hyperparameter tuning and cross-validation
- **Model Persistence**: Efficient serialization using joblib for production deployment
- **Label Encoding**: Robust categorical encoding for multi-class classification

### **Production Features**
- **RESTful API**: FastAPI-powered web service with automatic OpenAPI documentation
- **Real-time Inference**: Sub-second prediction latency with confidence scoring
- **Containerization**: Docker support with multi-stage builds for optimized deployment
- **CI/CD Pipeline**: Automated testing across Python versions with GitHub Actions
- **Comprehensive Testing**: Unit tests, integration tests, and API endpoint validation
- **Error Handling**: Robust exception handling with meaningful error responses

### **Key Capabilities**
- **Multi-class Classification**: 6 academic disciplines (Biology, Chemistry, CS, Mathematics, Medicine, Physics)
- **Confidence Scoring**: Probability-based confidence measures for predictions
- **Batch Processing**: Support for both single and batch inference
- **Model Evaluation**: Detailed performance metrics including accuracy, F1-score, and confusion matrices
- **Scalable Architecture**: Designed for horizontal scaling and cloud deployment

## Dataset Information

The project expects a CSV file with the following structure:
- `abstract`: Research abstract text
- `category`: Target classification category

Sample categories might include:
- Computer Science
- Biology 
- Physics
- Mathematics
- Chemistry
- Medicine

## Project Structure

```
research-abstract-classifier/
├── README.md
├── requirements.txt
├── Dockerfile
├── data/
│   └── sample_abstracts.csv
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── api/
│   └── app.py
├── tests/
│   └── test_pipeline.py
└── models/
    ├── classifier.pkl
    └── vectorizer.pkl
```

## Installation & Setup

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd research-abstract-classifier
```

2. **Create virtual environment**
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (first time only)
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## How to Run

### 1. Train the Model
```bash
python src/train.py
```

### 2. Evaluate Model Performance
```bash
python src/evaluate.py
```

### 3. Make Predictions
```bash
python src/inference.py --text "Your research abstract here"
```

### 4. Start the API Server
```bash
python api/app.py
```
The API will be available at `http://localhost:8000`

### 5. Run Tests
```bash
pytest tests/
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/
```

### Classify Abstract
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"abstract": "This study presents a novel deep learning approach for natural language processing tasks..."}'
```

Response:
```json
{
  "category": "Computer Science",
  "confidence": 0.89
}
```

## Docker Usage

### Build Image
```bash
docker build -t research-abstract-classifier .
```

### Run Container
```bash
docker run -p 8000:8000 research-abstract-classifier
```

### Using Docker Compose (Optional)
```bash
docker-compose up --build
```

## Model Performance

The trained model achieves:
- **Accuracy**: ~85% on test set
- **F1-Score**: ~0.83 (macro average)

Performance may vary depending on dataset quality and size.

## Development

### Adding New Categories
1. Update your training data with new category labels
2. Retrain the model: `python src/train.py`
3. Evaluate performance: `python src/evaluate.py`

### Improving Model Performance
- Increase training data size
- Experiment with different algorithms (SVM, Random Forest, etc.)
- Fine-tune hyperparameters
- Add feature engineering (n-grams, word embeddings)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
