# =========================================================
# Smart Farmer Advisory + Loan Risk System
# SAFE • FULL FEATURES • STREAMLIT CLOUD READY
# =========================================================

import os
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Farmer Advisory System",
    page_icon="🌾",
    layout="wide"
)

MODEL_FILE = "risk_model.pkl"
TARGET = "risk_flag"

# =========================================================
# OPEN-SOURCE WEATHER (OPEN-METEO)
# =========================================================
def get_lat_lon(city, state):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": f"{city}, {state}, India", "count": 1}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()
    if "results" in data:
        return data["results"][0]["latitude"], data["results"][0]["longitude"]
    return None, None


def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation",
    }
    r = requests.get(url, params=params, timeout=10)
    return r.json()["current"]

# =========================================================
# LOAD DATA (CSV / EXCEL / DEFAULT)
# =========================================================
def load_data(uploaded_file=None):
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)

    # DEFAULT DATA (SAFE FALLBACK)
    return pd.DataFrame({
        "state": ["Andhra Pradesh", "Karnataka", "Tamil Nadu", "Maharashtra"],
        "crop": ["Rice", "Maize", "Sugarcane", "Cotton"],
        "rainfall": [900, 750, 700, 850],
        "temperature": [32, 28, 33, 30],
        "risk_flag": [1, 0, 1, 0]
    })

# =========================================================
# UI
# =========================================================
st.title("🌾 Smart Farmer Advisory System")
st.caption("Climate • Crop • Risk • Loan Advisory (India)")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV / Excel (Optional)",
    type=["csv", "xlsx"]
)

state = st.sidebar.selectbox(
    "State",
    [
        "Andhra Pradesh", "Karnataka", "Tamil Nadu", "Maharashtra",
        "West Bengal", "Bihar", "Uttar Pradesh"
    ]
)

city = st.sidebar.text_input("Village / Town / City")

# =========================================================
# DATA
# =========================================================
df = load_data(uploaded_file)

st.subheader("📄 Data Preview")
st.dataframe(df)

# =========================================================
# WEATHER ADVISORY
# =========================================================
if city:
    lat, lon = get_lat_lon(city, state)
    if lat:
        weather = get_weather(lat, lon)
        st.success(
            f"🌦 Weather in {city}: "
            f"{weather['temperature_2m']}°C, "
            f"Rain: {weather['precipitation']} mm"
        )
    else:
        st.warning("Location not found")

# =========================================================
# MACHINE LEARNING (SAFE LOGIC)
# =========================================================
X = df.drop(columns=[TARGET], errors="ignore")
y = df[TARGET] if TARGET in df.columns else None

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols)
])

pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(n_estimators=150, random_state=42))
])

# TRAIN ONLY IF TARGET EXISTS
if y is not None:
    if not os.path.exists(MODEL_FILE):
        pipe.fit(X, y)
        joblib.dump(pipe, MODEL_FILE)
    else:
        pipe = joblib.load(MODEL_FILE)

# =========================================================
# PREDICTION
# =========================================================
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    df["predicted_risk"] = model.predict(X)

    st.subheader("🚨 Risk Prediction")
    st.dataframe(df)

    high_risk = df[df["predicted_risk"] == 1]
    if not high_risk.empty:
        st.error("⚠️ High Risk Areas")
        st.dataframe(high_risk)
    else:
        st.success("✅ Low Risk Overall")

# =========================================================
# DOWNLOAD
# =========================================================
st.download_button(
    "⬇️ Download Results (CSV)",
    df.to_csv(index=False),
    "farmer_advisory_results.csv",
    "text/csv"
)

# =========================================================
# LEGAL & SAFETY NOTE
# =========================================================
st.caption(
    "⚠️ Advisory system only. Not financial/legal advice. "
    "Uses open-source public data (Open-Meteo)."
)
