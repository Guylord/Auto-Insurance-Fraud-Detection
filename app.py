# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# Load raw dataset (optional cache function)
@st.cache_data
def load_raw_data(path="your_dataset.csv"):
    df = pd.read_csv(path)
    return df

# Load any pickle file (model or scaler)
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Preprocess data using pre-fitted scaler
def preprocess_data(df, scaler=None):
    # Handle missing and invalid values
    df['authorities_contacted'].fillna('Other', inplace=True)
    df['collision_type'].replace('?', "Other", inplace=True)
    for col in ['property_damage', 'police_report_available']:
        df[col] = df[col].replace('?', 'Not specified')

    # Convert date columns and calculate derived feature
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['days_since_bind'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['days_since_bind'] = df['days_since_bind'].fillna(-1)

    # Drop unused or high-cardinality columns
    cols_to_drop = [
        'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
        'incident_date', 'incident_state', 'incident_city', 'insured_hobbies',
        'auto_make', 'auto_model', 'auto_year', '_c39'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Encode target variable
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

    # Separate and transform features
    num_df = df.select_dtypes(include='number').drop(columns='fraud_reported')
    cat_df = pd.get_dummies(df.select_dtypes('object').drop(columns='policy_number'), drop_first=True)

    # Scale numerical features
    if scaler is None:
        scaler = StandardScaler()
        scaled_num_df = pd.DataFrame(scaler.fit_transform(num_df), columns=num_df.columns, index=num_df.index)
    else:
        scaled_num_df = pd.DataFrame(scaler.transform(num_df), columns=num_df.columns, index=num_df.index)

    # Combine features
    processed_df = pd.concat([scaled_num_df, cat_df], axis=1)

    # Reattach identifiers
    processed_df['policy_number'] = df['policy_number']
    processed_df['fraud_reported'] = df['fraud_reported']

    return processed_df

# Make prediction using a policy number
def make_prediction(model, preprocessed_df, policy_number):
    match = preprocessed_df[preprocessed_df['policy_number'].astype(str) == str(policy_number)]
    if match.empty:
        return None, "Policy number not found."

    X = match.drop(columns=["fraud_reported", "policy_number"])

    try:
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(X)[0][1]
        else:
            prediction = model.predict(X)[0]
    except Exception as e:
        return None, f"Prediction error: {e}"

    return prediction, None

# Load model and scaler
try:
    model = joblib.load('model_dt.joblib')
    scaler = joblib.load('scaler.joblib')

except FileNotFoundError as e:
    st.error(f"Model or scaler file not found: {e}")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")
st.title("🔍 Insurance Fraud Detection")

# File uploader
st.subheader("📂 Upload Insurance Claim CSV")
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        preprocessed_df = preprocess_data(df_raw, scaler=scaler)

        # Preview data
        st.subheader("📊 Preview Preprocessed Data")
        st.dataframe(preprocessed_df.head(50), use_container_width=True)

        # Prediction section
        st.subheader("🔎 Predict Fraud Probability")
        policy_number = st.text_input("Enter Policy Number")

        if st.button("Predict"):
            if not policy_number.strip():
                st.warning("Please enter a valid Policy Number.")
            else:
                prediction, error = make_prediction(model, preprocessed_df, policy_number)

                if error:
                    st.error(f"❌ {error}")
                else:
                    st.subheader("🧾 Prediction Result")
                    st.write(f"**Fraud Probability:** `{prediction:.4f}`")

                    if prediction > 0.5:
                        st.error("⚠️ High risk of fraud detected.")
                    else:
                        st.success("✅ Low risk of fraud.")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
