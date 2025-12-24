# =========================================================
# AGRICULTURAL LOAN DECISION SUPPORT SYSTEM (DSS)
# FINAL POLISHED VERSION – ORDER & COLOR LOCKED
# Safe | Legal | Educational | Visual | Impactful
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agri Loan Decision Support",
    page_icon="logo.png",
    layout="wide"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
    .main {background-color: #F9FFF9;}
    h1, h2, h3 {color: #2E7D32;}
    .stButton>button {background-color:#2E7D32; color:white; border-radius:8px;}
    .stDownloadButton>button {background-color:#1B5E20; color:white;}
    .css-1d391kg {background-color: #F1F8F4;}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 How to Use")
st.sidebar.markdown("""
1️⃣ Upload agricultural loan CSV  
2️⃣ View **risk insights**, not approval  
3️⃣ Read **improvement suggestions**  
4️⃣ Use visuals to understand patterns  

⚠️ Educational & awareness tool only
""")

st.sidebar.divider()
st.sidebar.info("📌 Decision Support Tool")

# ---------------- CSV FORMAT INFO ----------------
st.sidebar.subheader("📄 CSV Format Required")
st.sidebar.markdown("""
Your CSV must contain **exactly these column headers**:

