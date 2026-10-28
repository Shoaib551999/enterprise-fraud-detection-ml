from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(mean=3.5, sigma=1.2, size=n)
    hour = rng.integers(0, 24, size=n)
    customer_age = rng.integers(18, 81, size=n)
    account_age_days = rng.integers(1, 3651, size=n)
    previous_transactions = rng.integers(0, 501, size=n)
    is_international = rng.binomial(1, 0.15, size=n)

    # Artificial risk patterns used only for demonstration.
    risk_score = (
        0.7 * (amount > 200)
        + 0.6 * (hour < 5)
        + 0.8 * (account_age_days < 30)
        + 0.5 * (previous_transactions < 5)
        + 0.7 * is_international
    )

    # Calibrate the baseline to produce roughly 5% fraud.
    fraud_probability = 1 / (
        1 + np.exp(-(-4.2 + 1.2 * risk_score))
    )

    is_fraud = rng.binomial(1, fraud_probability)

    df = pd.DataFrame(
        {
            "transaction_id": [
                f"TXN-{i:06d}" for i in range(1, n + 1)
            ],
            "amount": np.round(amount, 2),
            "hour": hour,
            "customer_age": customer_age,
            "account_age_days": account_age_days,
            "previous_transactions": previous_transactions,
            "is_international": is_international,
            "is_fraud": is_fraud,
        }
    )

    return df


if __name__ == "__main__":
    output_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "transactions.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    transactions = generate_transactions()
    transactions.to_csv(output_path, index=False)

    print("Dataset generated successfully!")
    print(f"Total transactions: {len(transactions)}")
    print(f"Fraud transactions: {transactions['is_fraud'].sum()}")
    print(
        f"Fraud percentage: "
        f"{transactions['is_fraud'].mean() * 100:.2f}%"
    )
    print(f"Saved to: {output_path}")