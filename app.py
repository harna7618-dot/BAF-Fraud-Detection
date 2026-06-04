import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --------------------------------------------------
# Load Model Assets
# --------------------------------------------------

model = tf.keras.models.load_model("fraud_ann_model.keras")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --------------------------------------------------
# Streamlit Page
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Account Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Account Fraud Detection System")
st.markdown(
    """
    This application uses an Artificial Neural Network (ANN)
    trained on the Bank Account Fraud (BAF) dataset to estimate
    the probability of fraudulent banking activity.
    """
)

st.subheader("Customer Information")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    income = st.slider(
        "Income",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.1
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=10,
        max_value=90,
        value=30
    )

    credit_risk_score = st.number_input(
        "Credit Risk Score",
        min_value=-170,
        max_value=389,
        value=122
    )

    bank_months_count = st.number_input(
        "Bank Months Count",
        min_value=-1,
        max_value=32,
        value=5
    )

    proposed_credit_limit = st.number_input(
        "Proposed Credit Limit",
        min_value=190,
        max_value=2100,
        value=500
    )

    velocity_24h = st.number_input(
        "Velocity 24 Hours",
        min_value=1300.0,
        max_value=9507.0,
        value=4750.0
    )

with col2:

    session_length = st.number_input(
        "Session Length (Minutes)",
        min_value=0.0,
        max_value=86.0,
        value=5.0
    )

    payment_type = st.selectbox(
        "Payment Type",
        ["AA", "AB", "AC", "AD", "AE"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
    )

    housing_status = st.selectbox(
        "Housing Status",
        ["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
    )

    source = st.selectbox(
        "Source",
        ["INTERNET", "TELEAPP"]
    )

    device_os = st.selectbox(
        "Device OS",
        ["linux", "windows", "macintosh", "other", "x11"]
    )

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("Predict Fraud Risk"):

    # ----------------------------------------------
    # Create Base Input Record
    # ----------------------------------------------

    data = {
        'income': income,
        'name_email_similarity': 0.49,
        'prev_address_months_count': 12,
        'current_address_months_count': 120,
        'customer_age': customer_age,
        'days_since_request': 0.5,
        'intended_balcon_amount': 0.0,
        'zip_count_4w': 1000,
        'velocity_6h': 5000,
        'velocity_24h': velocity_24h,
        'velocity_4w': 5000,
        'bank_branch_count_8w': 10,
        'date_of_birth_distinct_emails_4w': 0,
        'credit_risk_score': credit_risk_score,
        'email_is_free': 1,
        'phone_home_valid': 1,
        'phone_mobile_valid': 1,
        'bank_months_count': bank_months_count,
        'has_other_cards': 0,
        'proposed_credit_limit': proposed_credit_limit,
        'foreign_request': 0,
        'session_length_in_minutes': session_length,
        'keep_alive_session': 1,
        'device_distinct_emails_8w': 1,
        'device_fraud_count': 0,
        'month': 6,
        'payment_type': payment_type,
        'employment_status': employment_status,
        'housing_status': housing_status,
        'source': source,
        'device_os': device_os
    }

    input_df = pd.DataFrame([data])

    # ----------------------------------------------
    # Apply One-Hot Encoding
    # ----------------------------------------------

    categorical_cols = [
        'payment_type',
        'employment_status',
        'housing_status',
        'source',
        'device_os'
    ]

    input_encoded = pd.get_dummies(
        input_df,
        columns=categorical_cols,
        drop_first=True
    )

    # ----------------------------------------------
    # Match Training Columns
    # ----------------------------------------------

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # ----------------------------------------------
    # Scaling
    # ----------------------------------------------

    input_scaled = scaler.transform(input_encoded)

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    fraud_probability = float(prediction[0][0])

    # ----------------------------------------------
    # Display Result
    # ----------------------------------------------

    st.subheader("Prediction Result")

    st.metric(
        label="Fraud Probability",
        value=f"{fraud_probability*100:.2f}%"
    )

    if fraud_probability >= 0.70:

        st.error(
            f"🚨 HIGH RISK ({fraud_probability*100:.2f}%)"
        )

    elif fraud_probability >= 0.30:

        st.warning(
            f"⚠️ MEDIUM RISK ({fraud_probability*100:.2f}%)"
        )

    else:

        st.success(
            f"✅ LOW RISK ({fraud_probability*100:.2f}%)"
        )