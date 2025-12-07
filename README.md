# 🔍 Insurance Fraud Detection App — Project Summary

A Streamlit web application built to detect fraudulent auto insurance claims using a pre-trained machine learning model. The app allows users to upload a CSV file, preprocesses the data automatically, and predicts fraud likelihood based on a supplied `policy_number`.

---

## 🚀 Core Functionality
- Upload and read insurance claims CSV
- Automated preprocessing pipeline:
  - Missing value treatment
  - '?' value replacement with meaningful labels
  - Date conversion + derived feature: `days_since_bind`
  - Dropping high-cardinality & multicollinear fields
  - One-hot encoding of categorical features
  - Numerical feature scaling using a pre-fitted StandardScaler
- View raw and processed dataset previews
- Predict fraud probability per policy record
- Displays features for selected policy before prediction
- Outputs probability scores with risk interpretation

---

## 🏗️ Prediction Flow
1. Upload dataset → `policy_number` sorted first  
2. Data is cleaned, encoded, scaled, and structured  
3. Select a policy number and run model inference  
4. App returns:
   - Feature snapshot of the selected policy
   - Fraud probability score
   - Risk assessment

---

## 📦 Tech Stack
| Component | Tool |
|----------|-------|
| Interface | Streamlit |
| Data Processing | Pandas, NumPy |
| Modeling | Scikit-Learn |
| Serialization | Joblib + Pickle |

---

