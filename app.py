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

    # Drop multicollinear features
    df.drop(columns = ['age', 'total_claim_amount'], inplace=True)

    # Convert date columns and calculate derived feature
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['days_since_bind'] = (df['incident_date'] - df['policy_bind_date']).dt.days

    # Drop unused or high-cardinality columns
    cols_to_drop = [
        'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
        'incident_date', 'incident_state', 'incident_city', 'insured_hobbies',
        'auto_make', 'auto_model', 'auto_year', '_c39'
    ]
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')

    df_clean['policy_number'] = df_clean['policy_number'].astype('object')

    # Encode target variable
    df_clean['fraud_reported'] = df_clean['fraud_reported'].map({'Y': 1, 'N': 0})

    # Separate and transform features
    X = df_clean.drop(columns = ['fraud_reported', 'policy_number'], axis=1)
    y = df_clean['fraud_reported']

    num_df = X.select_dtypes(include='number')
    cat_df = pd.get_dummies(X.select_dtypes('object'))

     # Scale numerical features
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(num_df)
    else:
        scaled = scaler.transform(num_df)

    scaled_num_df = pd.DataFrame(scaled, columns=num_df.columns, index=num_df.index)

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

        # Ensure 'policy_number' is the first column
        if 'policy_number' in df_raw.columns:
            cols = ['policy_number'] + [col for col in df_raw.columns if col != 'policy_number']
            df_raw = df_raw[cols]

        st.success("✅ File uploaded successfully!")

        
        # Preview raw data
        with st.expander("📄 Preview Raw Data"):
            st.dataframe(df_raw.head(50), use_container_width=True)

        # View features of raw data
        with st.expander("📌 View Raw Data Features"):
            st.write("**Columns in raw dataset:**")
            st.write(list(df_raw.columns))

        # Preprocess automatically and preview
        preprocessed_df = preprocess_data(df_raw, scaler=scaler)

        # Ensure 'policy_number' is first column in preprocessed data
        cols = ['policy_number'] + [col for col in preprocessed_df.columns if col not in ['policy_number']]
        preprocessed_df = preprocessed_df[cols]

        st.session_state['preprocessed_df'] = preprocessed_df

        with st.expander("📊 Preview Preprocessed Data"):
            st.dataframe(preprocessed_df.head(50), use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# Predict section (enabled only if preprocessing is done)
if 'preprocessed_df' in st.session_state:
    st.subheader("🔎 Predict Fraud Probability")
    policy_number = st.text_input("Enter Policy Number")

    if st.button("🔍 Predict"):
        if not policy_number.strip():
            st.warning("Please enter a valid Policy Number.")
        else:
            preprocessed_df = st.session_state['preprocessed_df']
            match = preprocessed_df[preprocessed_df['policy_number'].astype(str) == str(policy_number)]

            if match.empty:
                st.error("❌ Policy number not found.")
            else:
                st.subheader("📌 Features of Selected Policy Number")
                st.dataframe(match.drop(columns=['fraud_reported'], errors='ignore').T.rename(columns={match.index[0]: 'Value'}), use_container_width=True)

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
