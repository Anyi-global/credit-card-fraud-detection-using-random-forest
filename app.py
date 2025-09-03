import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load pipeline
pipeline = joblib.load("fraud_pipeline.joblib")

# --------- Streamlit Page Setup ----------
st.set_page_config(page_title="Fraud Detection Model", layout="centered")

st.markdown("""
    <style>
        body, .stApp {
            background-color: #000000;
            color: white;
        }

        /* Make form label text white */
        label, .stTextInput label, .stSelectbox label, .stNumberInput label, .stSlider label {
            color: white !important;
        }

        /* Subheader and title */
        h1, h2, h3, .stMarkdown, .css-1v0mbdj, .css-1d391kg {
            color: white !important;
        }

        /* Make button text and background more visible */
        .stButton>button {
            background-color: #1a1a1a !important;
            color: white !important;
            border: 1px solid white;
        }

        .stButton>button:hover {
            background-color: #333333 !important;
            color: #00ffcc !important;
            border: 1px solid #00ffcc;
        }

    </style>
""", unsafe_allow_html=True)

st.title("Fraud Detection App")
st.write("Enter transaction details or use the default values (from test data).")

# Loading some test data for defaults (replace with your actual test set)
test_data = pd.read_csv("creditcard.csv")  # or load your prepared X_test
defaults = test_data.iloc[0][1:-1].values  # skip Time (col 0) and Class (last col)

inputs = []
for i in range(1, 29):  # V1 to V28
    val = st.number_input(f"V{i}", value=float(defaults[i-1]), format="%.6f")
    inputs.append(val)

# Add Amount (last feature before Class)
amount = st.number_input("Transaction Amount", value=float(test_data.iloc[0]['Amount']), format="%.2f")
inputs.append(amount)

# Convert input into numpy array
inputs = np.array(inputs).reshape(1, -1)

# Prediction
if st.button("Check Transaction"):
    prediction = pipeline.predict(inputs)[0]
    proba = pipeline.predict_proba(inputs)[0][1]  # fraud probability
    
    if prediction == 1:
        st.error(f"⚠️ This transaction is predicted as **FRAUD** (Probability: {proba:.2f})")
    else:
        st.success(f"✅ This transaction is predicted as **NON-FRAUD** (Probability: {proba:.2f})")
