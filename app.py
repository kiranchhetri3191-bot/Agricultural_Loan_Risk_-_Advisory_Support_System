# app.py
# =========================================================
# AGRICULTURAL LOAN VISUAL ANALYTICS SYSTEM
# Decision Support | Legal | Educational | Demo Only
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agri Loan Visual Analytics",
    page_icon="🌾",
    layout="wide"
)

# ---------------- DISCLAIMER ----------------
st.markdown("""
## ⚠️ LEGAL DISCLAIMER
This application is a **Decision Support System (DSS)** for:
- Education
- Research
- Farmer awareness

❌ Not a bank / NBFC system  
❌ Not RBI compliant decision engine  
❌ No real farmer or credit bureau data used  

All outputs are **simulated insights**, not approvals.
""")

st.divider()

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; color:#2E8B57;'>🌾 Agricultural Loan Risk & Insight System</h1>
<p style='text-align:center;'>CSV Upload • Visual Insights • Real-Life Learning</p>
""", unsafe_allow_html=True)

# ---------------- DEMO DATA ----------------
def generate_demo_data(n=6000):
    crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"]
    irrigation = ["Rainfed", "Canal", "Borewell"]

    df = pd.DataFrame({
        "farmer_age": np.random.randint(21, 65, n),
        "land_size_acres": np.round(np.random.uniform(0.5, 10, n), 2),
        "annual_farm_income": np.random.randint(120000, 900000, n),
        "loan_amount": np.random.randint(50000, 500000, n),
        "crop_type": np.random.choice(crops, n),
        "irrigation_type": np.random.choice(irrigation, n),
        "existing_loans": np.random.randint(0, 3, n),
        "credit_score": np.random.randint(300, 850, n)
    })

    df["loan_approved"] = np.where(
        (df["credit_score"] >= 650) &
        (df["annual_farm_income"] >= df["loan_amount"] * 1.3) &
        (df["land_size_acres"] >= 1) &
        (df["existing_loans"] <= 1),
        1, 0
    )
    return df

# ---------------- TRAIN MODEL ----------------
@st.cache_data
def train_model():
    data = generate_demo_data()

    le_crop = LabelEncoder()
    le_irrigation = LabelEncoder()

    data["crop_type"] = le_crop.fit_transform(data["crop_type"])
    data["irrigation_type"] = le_irrigation.fit_transform(data["irrigation_type"])

    X = data.drop("loan_approved", axis=1)
    y = data["loan_approved"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, le_crop, le_irrigation

model, le_crop, le_irrigation = train_model()

# ---------------- CSV UPLOAD ----------------
st.sidebar.header("📤 Upload Agricultural Loan CSV")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = [
        "farmer_age","land_size_acres","annual_farm_income",
        "loan_amount","crop_type","irrigation_type",
        "existing_loans","credit_score"
    ]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ Invalid CSV format")
        st.stop()

    df["crop_type"] = le_crop.transform(df["crop_type"])
    df["irrigation_type"] = le_irrigation.transform(df["irrigation_type"])

    df["Approval"] = model.predict(df)
    df["Approval"] = df["Approval"].map({1: "Approved (Demo)", 0: "Rejected (Demo)"})

    st.success("✅ Analysis Completed")

    st.dataframe(df)

    st.divider()

    # ---------------- VISUALIZATION SECTION ----------------
    st.header("📊 Visual Insights")

    col1, col2 = st.columns(2)

    # Pie Chart
    with col1:
        st.subheader("Loan Approval Distribution")
        fig, ax = plt.subplots()
        df["Approval"].value_counts().plot.pie(
            autopct='%1.1f%%', ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    # Credit Score Histogram
    with col2:
        st.subheader("Credit Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["credit_score"], bins=20)
        ax.set_xlabel("Credit Score")
        ax.set_ylabel("Farmers")
        st.pyplot(fig)

    # Scatter Plot
    st.subheader("Income vs Loan Amount (Risk View)")
    fig, ax = plt.subplots()
    ax.scatter(df["annual_farm_income"], df["loan_amount"])
    ax.set_xlabel("Annual Farm Income")
    ax.set_ylabel("Loan Amount")
    st.pyplot(fig)

    # Crop-wise approval
    st.subheader("Crop-wise Loan Approval Count")
    crop_chart = df.groupby("crop_type")["Approval"].value_counts().unstack().fillna(0)
    fig, ax = plt.subplots()
    crop_chart.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Irrigation impact
    st.subheader("Irrigation Impact on Loan Approval")
    irrig_chart = df.groupby("irrigation_type")["Approval"].value_counts().unstack().fillna(0)
    fig, ax = plt.subplots()
    irrig_chart.plot(kind="bar", ax=ax)
    st.pyplot(fig)

else:
    st.info("📌 Upload agricultural loan CSV to see insights")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
### 🌱 Why This Matters
✔ Farmers understand risk factors  
✔ NGOs identify support gaps  
✔ Students build real-world finance analytics  
✔ Policy impact without legal exposure  
""")
