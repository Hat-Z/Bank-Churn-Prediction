import streamlit as st
import pandas as pd
import joblib

# Load model pipeline
model = joblib.load("churn_model.pkl")

st.title("🏦 Bank Customer Churn Prediction")

st.write("Fill in the customer details below to predict if they are likely to leave the bank.")

# User input
credit_score = st.number_input("Credit Score", 300, 900, 650)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
products_number = st.slider("Number of Products", 1, 4, 1)
credit_card = st.selectbox("Has Credit Card?", [0, 1])
active_member = st.selectbox("Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Convert to dataframe
input_data = pd.DataFrame([{
    "credit_score": credit_score,
    "country": country,
    "gender": gender,
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": products_number,
    "credit_card": credit_card,
    "active_member": active_member,
    "estimated_salary": estimated_salary
}])

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of churn

    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to leave** the bank. (Churn probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is **likely to stay** with the bank. (Churn probability: {probability:.2f})")
