import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏡 House Price Predictor")

# Load model artifacts
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/feature_columns.pkl")

# UI Inputs (a few key ones for simplicity)
st.sidebar.header("Enter House Details")

GrLivArea = st.sidebar.slider("Above Ground Living Area (sqft)", 300, 5000, 1500)
OverallQual = st.sidebar.selectbox("Overall Quality (1-10)", list(range(1, 11)), index=5)
GarageCars = st.sidebar.slider("Garage Capacity (cars)", 0, 4, 2)
TotalBsmtSF = st.sidebar.slider("Total Basement Area (sqft)", 0, 3000, 800)
FullBath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
YearBuilt = st.sidebar.slider("Year Built", 1900, 2025, 2000)

# Create a dummy dataframe with default values
input_dict = {
    "GrLivArea": GrLivArea,
    "OverallQual": OverallQual,
    "GarageCars": GarageCars,
    "TotalBsmtSF": TotalBsmtSF,
    "FullBath": FullBath,
    "YearBuilt": YearBuilt
}

df_input = pd.DataFrame([input_dict])
df_input = df_input.reindex(columns=columns, fill_value=0)  # align with training features
df_input_scaled = scaler.transform(df_input)

if st.button("Predict Price"):
    prediction = model.predict(df_input_scaled)[0]
    st.success(f"💰 Estimated Sale Price: ${prediction:,.0f}")
