# Insurance Fraud Detection Web Application

This project provides an end-to-end machine learning solution designed to identify potential auto insurance fraud using historical claim records. The system is deployed as an interactive **Streamlit web application**, enabling real-time fraud probability scoring through an intuitive user interface. The model takes a unique `policy_number` as input, retrieves associated features from the dataset, preprocesses the data, and generates a probability score indicating the likelihood of fraudulent activity.

---

## Overview

Insurance fraud remains one of the largest financial burdens in the insurance industry. This application demonstrates a practical approach to fraud risk assessment by integrating machine learning with automated data cleaning, feature engineering, and live prediction capabilities. Users can upload raw claim files, explore the dataset, and run fraud detection against any specific policy in seconds.

The primary goal of the project is to build an easily deployable, user-friendly fraud prediction engine suitable for risk assessment teams, auditors, and insurance analysts.

---

## Key Features

### 1. Data Upload & Validation
- Accepts CSV claim datasets through the UI.
- Automatically reorders features to ensure `policy_number` appears first.

### 2. Intelligent Preprocessing Pipeline
The application includes a built-in transformation workflow that:
- Handles missing values and replaces ambiguous entries such as `'?'`.
- Converts date fields and creates a derived feature: `days_since_bind`.
- Drops multicollinear, noisy, and high-cardinality features.
- Applies one-hot encoding to categorical variables.
- Scales numerical columns using a pre-trained `StandardScaler`.

### 3. Fraud Prediction Engine
- Powered by a trained machine learning model loaded via Joblib.
- Generates probability output for fraud likelihood.
- Policy-specific feature breakdown is displayed before prediction.
- Interactive decision feedback:
  - `> 0.5` → **High risk of fraud**
  - `≤ 0.5` → **Low fraud likelihood**

---

## System Workflow

1. **Upload dataset**
2. **Preprocessing executes automatically**
3. **User enters a policy_number**
4. App extracts and prepares features for only that policy
5. ML model predicts and returns fraud probability
6. The result is displayed with interpretation & risk score

---

## Technology Stack

| Layer | Tools/Packages Used |
|-------|---------------------|
| Interface & Deployment | Streamlit |
| Data Processing | Pandas, NumPy |
| Modeling & Scaling | Scikit-Learn |
| Serialization | Joblib & Pickle |
| Hosting | Streamlit Cloud |

## 🔗 Live Application
The system is publicly available online for instant usage:

**https://auto-insurance-fraud-detection.streamlit.app/**

---
streamlit run app.py
