from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "transactions.csv"
)


def explore_dataset():
    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET OVERVIEW =====")
    print(f"Total transactions: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\n===== FIRST FIVE TRANSACTIONS =====")
    print(df.head().to_string(index=False))

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATE TRANSACTION IDs =====")
    print(df["transaction_id"].duplicated().sum())

    print("\n===== FRAUD DISTRIBUTION =====")
    print(df["is_fraud"].value_counts())

    print("\n===== FRAUD PERCENTAGE =====")
    print(f"{df['is_fraud'].mean() * 100:.2f}%")

    print("\n===== TRANSACTION AMOUNT STATISTICS =====")
    print(df["amount"].describe())


if __name__ == "__main__":
    explore_dataset()