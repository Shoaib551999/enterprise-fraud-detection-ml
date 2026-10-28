from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"

FEATURES = [
    "amount",
    "hour",
    "customer_age",
    "account_age_days",
    "previous_transactions",
    "is_international",
]


def train_random_forest():

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Random Forest fraud detection model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, predictions))

    print("\n===== ROC-AUC SCORE =====")
    print(f"{roc_auc_score(y_test, probabilities):.4f}")

    print("\n===== PR-AUC SCORE =====")
    print(f"{average_precision_score(y_test, probabilities):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully: {MODEL_PATH}")


if __name__ == "__main__":
    train_random_forest()