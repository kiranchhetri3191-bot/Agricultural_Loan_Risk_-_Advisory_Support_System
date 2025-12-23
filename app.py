# app.py
# Smart Farmer Advisory System (SAFE VERSION)

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Smart Farmer Advisory System", layout="wide")

MODEL_FILE = "risk_model.pkl"
TARGET = "risk_flag"

# -------------------------------------------------
# SAMPLE DATA (replace later with API / CSV)
# -------------------------------------------------
def load_data():
    data = {
        "state": ["Andhra Pradesh", "Andhra Pradesh", "Karnataka", "Tamil Nadu"],
        "rainfall": [900, 1200, 800, 700],
        "temperature": [32, 30, 28, 33],
        "crop": ["Rice", "Cotton", "Maize", "Sugarcane"],
        "risk_flag": [1, 0, 1, 0],  # target
    }
    return pd.DataFrame(data)

df = load_data()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌾 Smart Farmer Advisory System")

st.sidebar.header("📍 Location Advisory (India)")
state = st.sidebar.selectbox("State", sorted(df["state"].unique()))
city = st.sidebar.text_input("Search Village / Town / City")

st.divider()

# -------------------------------------------------
# DEBUG (remove later)
# -------------------------------------------------
st.write("🔎 Current columns:", df.columns.tolist())

# -------------------------------------------------
# FEATURE / TARGET SPLIT (SAFE)
# -------------------------------------------------
X = df.drop(columns=[TARGET], errors="ignore")

y = df[TARGET] if TARGET in df.columns else None

# -------------------------------------------------
# PREPROCESSING
# -------------------------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# -------------------------------------------------
# TRAIN MODEL (ONLY IF NEEDED)
# -------------------------------------------------
if y is not None:
    if not os.path.exists(MODEL_FILE):
        pipe.fit(X, y)
        joblib.dump(pipe, MODEL_FILE)
        st.success("✅ Model trained and saved")
    else:
        pipe = joblib.load(MODEL_FILE)
        st.info("📦 Model loaded from disk")
else:
    st.warning("⚠️ Target variable not found. Model not trained.")

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------
st.subheader("📊 Risk Prediction")

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)

    preds = model.predict(X)
    df["predicted_risk"] = preds

    st.dataframe(df)

    high_risk = df[df["predicted_risk"] == 1]
    if not high_risk.empty:
        st.error("🚨 High Risk Areas Detected")
        st.dataframe(high_risk)
    else:
        st.success("✅ No high-risk areas detected")

else:
    st.info("Model not available yet.")
