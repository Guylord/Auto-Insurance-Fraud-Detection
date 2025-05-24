# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = tf.keras.models.load_model("best_fraud_model.h5")
scaler = joblib.load("scaler.pkl")

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("your_dataset.csv")

    # Fill missing values
    df['authorities_contacted'].fillna('Other', inplace=True)
    df['collision_type'].replace('?', "Other", inplace=True)
    for col in ['property_damage', 'police_report_available']:
        df[col] = df[col].replace('?', 'Not specified')

    # Dates and days difference
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['days_since_bind'] = (df['incident_date'] - df['policy_bind_date']).dt.days

    # Drop unused columns
    cols_to_drop = [
        'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
        'incident_date', 'incident_state', 'incident_city', 'insured_hobbies',
        'auto_make', 'auto_model', 'auto_year', '_c39'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Encode target
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

    # One-hot encoding
    cat_df = pd.get_dummies(df.select_dtypes('object').drop(columns='policy_number'), drop_first=True)
    num_df = df.select_dtypes(include='number').drop(columns='fraud_reported')

    # Combine numerical and categorical
    processed_df = pd.concat([num_df, cat_df], axis=1)

    # Attach policy_number and fraud_reported back for reference
    processed_df['policy_number'] = df['policy_number']
    processed_df['fraud_reported'] = df['fraud_reported']

    return processed_df

data = load_data()

# Streamlit app layout
st.set_page_config(page_title="Insurance Fraud Prediction", layout="centered")
st.title("🔍 Insurance Fraud Detection")
st.markdown("Enter the **Policy Number** to predict the probability of fraud.")

# Input field for policy number
policy_number = st.text_input("Policy Number", max_chars=50)

if st.button("Predict"):
    if policy_number not in data['policy_number'].astype(str).values:
        st.error("❌ Policy number not found in dataset.")
    else:
        # Extract preprocessed features
        match = data[data['policy_number'].astype(str) == policy_number]
        X = match.drop(columns=["fraud_reported", "policy_number"])
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0][0]

        # Output
        st.subheader("Prediction Result")
        st.write(f"**Fraud Probability:** `{prediction:.4f}`")

        if prediction > 0.5:
            st.error("⚠️ High risk of fraud detected.")
        else:
            st.success("✅ Low risk of fraud.")
