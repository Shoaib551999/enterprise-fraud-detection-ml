# Enterprise Fraud Detection ML Platform

An end-to-end machine learning engineering project demonstrating fraud detection, model evaluation, experiment tracking, REST API serving, containerization, and automated testing.

**Disclaimer:** This is a portfolio demonstration built using synthetic transaction data. The models are not suitable for real financial fraud decisions.

## Project Overview

This project demonstrates how to build a machine learning platform that evaluates transaction fraud risk and exposes predictions through an API and interactive dashboard.

The platform includes:

- Synthetic transaction data generation
- Exploratory data analysis
- Logistic Regression and Random Forest models
- Model comparison and threshold optimization
- MLflow experiment tracking
- FastAPI model serving
- Streamlit dashboard
- Docker containerization
- Automated API testing with Pytest
- GitHub Actions continuous integration configuration

## System Architecture

```text
Synthetic Transaction Generator
              |
              v
       Transaction Dataset
              |
              v
      Model Training Pipeline
              |
       +------+------+
       |             |
       v             v
Logistic Regression  Random Forest
       |             |
       +------+------+
              |
              v
   Validation & Model Selection
              |
              v
      Saved Model Artifact
              |
              v
         FastAPI Service
              |
       +------+------+
       |             |
       v             v
 Streamlit UI     REST Clients
```

MLflow separately records model experiments, metrics, and model artifacts.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| API Development | FastAPI, Pydantic |
| Dashboard | Streamlit |
| Containerization | Docker |
| Testing | Pytest, HTTPX |
| CI | GitHub Actions |

## Machine Learning Approach

The project trains and compares two classification algorithms:

1. Logistic Regression
2. Random Forest

The dataset is split into training (60%), validation (20%), and test (20%) subsets.

Model and classification-threshold selection use validation data. Final evaluation is performed on the held-out test partition.

### Selected Model

**Random Forest**

| Metric | Result |
|---|---:|
| Classification threshold | 0.40 |
| Test precision | 0.0474 |
| Test recall | 0.2553 |
| Test F1-score | 0.0800 |
| Test PR-AUC | 0.0500 |
| Test ROC-AUC | 0.6289 |

The model identified 12 of 47 fraudulent transactions in the test set, while incorrectly flagging 241 legitimate transactions.

These results demonstrate the challenges of class imbalance, noisy synthetic labels, and fraud detection tradeoffs. They do not establish real-world fraud detection effectiveness.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Fraud risk prediction |

### Example Prediction Request

```json
{
  "amount": 450,
  "hour": 2,
  "customer_age": 25,
  "account_age_days": 10,
  "previous_transactions": 2,
  "is_international": 1
}
```

### Example Prediction Response

```json
{
  "fraud_probability": 0.3646,
  "is_fraud": false,
  "risk_classification": "Not flagged",
  "threshold": 0.4,
  "model_name": "Random Forest"
}
```

The API returns the model's estimated fraud probability and determines whether a transaction should be flagged for review using the selected threshold.

## Running the API Locally

Create and activate a Python virtual environment, then install the API dependencies:

```bash
python -m pip install -r requirements-api.txt
```

Start the FastAPI server:

```bash
python -m uvicorn src.api:app --reload --port 8000
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

The saved model artifact must exist at `models/best_fraud_model.joblib`.

## Running the Streamlit Dashboard

Install the dashboard dependencies:

```bash
python -m pip install streamlit requests
```

With the FastAPI service running on port 8000, start Streamlit:

```bash
python -m streamlit run src/dashboard.py
```

The dashboard accepts transaction details and displays the model's prediction.

## Running with Docker

Build the Docker image:

```bash
docker build -t fraud-detection-api .
```

Run the container:

```bash
docker run -d --name fraud-api -p 8001:8000 fraud-detection-api
```

The Dockerized API documentation is available at `http://127.0.0.1:8001/docs`.

## MLflow Experiment Tracking

The project uses MLflow to record machine learning experiments.

Install MLflow:

```bash
python -m pip install mlflow
```

Start the tracking server:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

In a separate terminal, run:

```bash
python src/track_experiments.py
```

The experiment is named **Enterprise Fraud Detection**.

## Automated Testing

Install testing dependencies:

```bash
python -m pip install pytest httpx
```

Run the API tests:

```bash
python -m pytest tests/test_api.py -v
```

The test suite covers API health, successful prediction responses, and invalid input validation.

## Continuous Integration

The repository includes a GitHub Actions workflow at:

`.github/workflows/tests.yml`

The workflow is configured to install dependencies and execute automated API tests on pushes and pull requests.

## Engineering Considerations

This project demonstrates several practical machine learning engineering concepts:

- Handling imbalanced classification datasets
- Comparing models using precision, recall, F1-score, ROC-AUC, and PR-AUC
- Selecting classification thresholds using validation data
- Separating model training from API inference
- Tracking experiments and artifacts with MLflow
- Validating API inputs using Pydantic
- Packaging model-serving applications with Docker
- Automating regression tests

## Limitations and Future Improvements

The current platform uses synthetic transaction data and a relatively small set of features.

Future improvements could include probability calibration, cost-sensitive threshold selection, feature engineering, drift monitoring, model registry integration, and evaluation using appropriately authorized real-world datasets.

The platform is intended to demonstrate engineering workflows rather than claim production fraud detection performance.