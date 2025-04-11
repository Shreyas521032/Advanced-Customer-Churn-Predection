import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
joblib.dump(model, 'Gradient_Boosting_Classifier.joblib')

# Load the trained model and scaler
model = joblib.load('saved_models/Gradient_Boosting_Classifier.joblib')
scaler = joblib.load('saved_models/scaler.joblib')

st.title("Customer Churn Prediction")

# Input fields for user data
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
# Add more input fields as per your model's requirements

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame([user_input], columns=expected_columns)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.write(f"Prediction: {result}")
