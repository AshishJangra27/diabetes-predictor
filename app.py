import streamlit as st
import joblib
import numpy as np

st.title("Diabetes Progression Predictor")

# Load model
model = joblib.load("model.pkl")

# Feature inputs
st.subheader("Enter Patient Features (10 inputs)")
features = []
feature_names = [
    "Age", "Sex", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"
]

for name in feature_names:
    val = st.number_input(f"{name}", value=0.0)
    features.append(val)

# Prediction
if st.button("Predict Progression"):
    prediction = model.predict([features])
    st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")
