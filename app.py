# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
# Load saved model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
st.title("Bank Churn Prediction App")
st.write("Enter customer details to predict if they might leave the bank.")
# --- User inputs ---
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=15, value=5)
products_number = st.number_input("Number of Bank Products", min_value=1, max_value=10, value=1)
credit_card = st.selectbox("Has Credit Card?", [0, 1])
active_member = st.selectbox("Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
gender = st.selectbox("Gender", ["Male", "Female"])
country = st.selectbox("Country", ["France", "Spain", "Germany"])
# --- Create DataFrame for model input ---
input_data = pd.DataFrame({
    'credit_score': [credit_score],
    'age': [age],
    'balance': [balance],
    'tenure': [tenure],
    'products_number': [products_number],
    'credit_card': [credit_card],
    'active_member': [active_member],
    'estimated_salary': [estimated_salary],
    'gender_Male': [1 if gender=="Male" else 0],
    'country_Spain': [1 if country=="Spain" else 0],
    'country_France': [1 if country=="France" else 0]
})
# --- Scale features ---
input_scaled = scaler.transform(input_data)
# --- Prediction ---
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]
# --- Display results ---
st.subheader("Prediction Results")
st.write("Churn Prediction:", "**Yes**" if prediction == 1 else "**No**")
st.write(f"Churn Probability: {probability:.2f}")
