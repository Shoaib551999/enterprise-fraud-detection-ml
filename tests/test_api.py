import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


# Test 1: Verify API homepage
def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()
    assert "model" in response.json()


# Test 2: Verify health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test 3: Verify fraud prediction
def test_valid_prediction():
    transaction = {
        "amount": 450,
        "hour": 2,
        "customer_age": 25,
        "account_age_days": 10,
        "previous_transactions": 2,
        "is_international": 1,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["fraud_probability"] <= 1
    assert isinstance(result["is_fraud"], bool)
    assert "risk_classification" in result
    assert "threshold" in result
    assert "model_name" in result

    assert result["is_fraud"] == (
        result["fraud_probability"] >= result["threshold"]
    )


# Test 4: Reject negative transaction amounts
def test_negative_amount():
    transaction = {
        "amount": -100,
        "hour": 2,
        "customer_age": 25,
        "account_age_days": 10,
        "previous_transactions": 2,
        "is_international": 1,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


# Test 5: Reject invalid transaction hours
def test_invalid_hour():
    transaction = {
        "amount": 450,
        "hour": 30,
        "customer_age": 25,
        "account_age_days": 10,
        "previous_transactions": 2,
        "is_international": 1,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422


# Test 6: Reject missing required fields
def test_missing_fields():
    transaction = {
        "amount": 450,
        "hour": 2,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422