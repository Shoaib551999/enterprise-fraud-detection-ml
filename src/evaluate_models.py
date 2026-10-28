from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.joblib"

FEATURES = [
    "amount",
    "hour",
    "customer_age",
    "account_age_days",
    "previous_transactions",
    "is_international",
]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df["is_fraud"]

# 60% training, 20% validation, 20% testing
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

numeric_features = [
    "amount",
    "hour",
    "customer_age",
    "account_age_days",
    "previous_transactions",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        ("binary", "passthrough", ["is_international"]),
    ]
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

best_f1 = -1
best_model = None
best_name = None
best_threshold = None

print("\n===== VALIDATION RESULTS =====")

for name, model in models.items():
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(y_val, probabilities)
    roc_auc = roc_auc_score(y_val, probabilities)

    for threshold in np.arange(0.05, 0.96, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        f1 = f1_score(y_val, predictions, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            best_threshold = float(threshold)

    print(
        f"{name}: PR-AUC={pr_auc:.4f}, "
        f"ROC-AUC={roc_auc:.4f}"
    )

print("\n===== SELECTED MODEL =====")
print(f"Model: {best_name}")
print(f"Threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_f1:.4f}")

# Evaluate once on the untouched test set
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n===== FINAL TEST RESULTS =====")
print(
    "Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "F1:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "PR-AUC:",
    round(average_precision_score(y_test, test_probabilities), 4)
)
print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, test_probabilities), 4)
)
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "threshold": best_threshold,
        "features": FEATURES,
        "model_name": best_name,
    },
    MODEL_PATH,
)

print(f"\nBest model saved to: {MODEL_PATH}")