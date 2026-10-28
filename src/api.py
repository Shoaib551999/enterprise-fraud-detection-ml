from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.joblib"

# Load the locally trained model and its selected threshold.
# Only load model files from trusted sources.
model_artifact = joblib.load(MODEL_PATH)

model = model_artifact["model"]
threshold = model_artifact["threshold"]
features = model_artifact["features"]

app = FastAPI(
    title="Enterprise Fraud Detection API",
    description="ML-powered fraud risk prediction using synthetic transaction data",
    version="1.0.0",
)


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    customer_age: int = Field(ge=18, le=120)
    account_age_days: int = Field(ge=0)
    previous_transactions: int = Field(ge=0)
    is_international: int = Field(ge=0, le=1)


@app.get("/")
def home():
    return {
        "message": "Enterprise Fraud Detection API is running",
        "model": model_artifact["model_name"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction):
    transaction_data = pd.DataFrame(
        [transaction.model_dump()]
    )

    transaction_data = transaction_data[features]

    fraud_probability = float(
        model.predict_proba(transaction_data)[0][1]
    )

    is_fraud = fraud_probability >= threshold

    return {
        "fraud_probability": round(fraud_probability, 4),
        "is_fraud": bool(is_fraud),
        "risk_classification": (
            "Flag for review" if is_fraud else "Not flagged"
        ),
        "threshold": round(float(threshold), 4),
        "model_name": model_artifact["model_name"],
    }