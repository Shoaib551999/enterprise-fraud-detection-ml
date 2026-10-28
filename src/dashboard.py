import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Enterprise Fraud Detection")
st.caption("Machine Learning Fraud Risk Assessment Dashboard")

st.info(
    "Demo only: This model was trained on synthetic data. "
    "Predictions are not suitable for real financial decisions."
)

st.divider()

st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=450.0,
        step=10.0,
    )

    hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=2,
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=120,
        value=25,
    )

with col2:
    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=0,
        value=10,
    )

    previous_transactions = st.number_input(
        "Previous Transactions",
        min_value=0,
        value=2,
    )

    is_international = st.selectbox(
        "International Transaction?",
        options=["No", "Yes"],
        index=1,
    )

transaction = {
    "amount": amount,
    "hour": hour,
    "customer_age": customer_age,
    "account_age_days": account_age_days,
    "previous_transactions": previous_transactions,
    "is_international": 1 if is_international == "Yes" else 0,
}

st.divider()

if st.button("🔍 Predict Fraud Risk", type="primary"):

    try:
        response = requests.post(
            API_URL,
            json=transaction,
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Results")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Fraud Probability",
                f"{result['fraud_probability'] * 100:.2f}%",
            )

        with metric2:
            st.metric(
                "Classification Threshold",
                f"{result['threshold'] * 100:.0f}%",
            )

        with metric3:
            st.metric(
                "Model",
                result["model_name"],
            )

        if result["is_fraud"]:
            st.warning(
                "⚠️ Transaction flagged for review. "
                "The model detected elevated fraud risk."
            )
        else:
            st.success(
                "✅ Transaction not flagged by the model."
            )

        st.subheader("Risk Visualization")

        st.progress(
            min(max(result["fraud_probability"], 0.0), 1.0)
        )

        with st.expander("View API Response"):
            st.json(result)

    except requests.exceptions.RequestException as error:
        st.error(
            f"Unable to connect to the Fraud Detection API: {error}"
        )

        st.info(
            "Make sure FastAPI is running on port 8000."
        )