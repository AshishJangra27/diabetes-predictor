import streamlit as st
import joblib
import numpy as np

st.title("Diabetes Progression Predictor")

# Load model
model = joblib.load("model.pkl")

st.subheader("Enter Patient Features as a Python list")
user_input = st.text_input("Example: [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]")

if st.button("Predict Progression"):
    try:
        # Evaluate string to list
        features = eval(user_input)
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Invalid input: {e}")
