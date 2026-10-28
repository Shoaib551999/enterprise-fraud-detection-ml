from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

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
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "transactions.csv"

FEATURES = [
    "amount",
    "hour",
    "customer_age",
    "account_age_days",
    "previous_transactions",
    "is_international",
]

# Connect to the MLflow server running in your other terminal.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Enterprise Fraud Detection")

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["is_fraud"]

# Match the training and validation splits from Step 9.
X_train_val, _, y_train_val, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            [
                "amount",
                "hour",
                "customer_age",
                "account_age_days",
                "previous_transactions",
            ],
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

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\nTraining and tracking: {name}")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = model.predict(X_val)

        mlflow.log_param("model_name", name)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("classification_threshold", 0.5)

        mlflow.log_metric(
            "precision",
            precision_score(y_val, predictions, zero_division=0),
        )
        mlflow.log_metric(
            "recall",
            recall_score(y_val, predictions, zero_division=0),
        )
        mlflow.log_metric(
            "f1",
            f1_score(y_val, predictions, zero_division=0),
        )
        mlflow.log_metric(
            "pr_auc",
            average_precision_score(y_val, probabilities),
        )
        mlflow.log_metric(
            "roc_auc",
            roc_auc_score(y_val, probabilities),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="fraud_model",
            skops_trusted_types=["numpy.dtype","sklearn.tree._tree.Tree"],
        )

        print(f"Successfully tracked: {name}")

print("\nBoth experiments logged successfully!")