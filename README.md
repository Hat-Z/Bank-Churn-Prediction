# 🏦 Bank Customer Churn Prediction

This project predicts whether a bank customer is likely to leave (churn) or stay based on their demographic and financial information.

## 🔍 Overview
The model uses machine learning (Random Forest Classifier) and a preprocessing pipeline that includes:
- OneHotEncoding for categorical features
- StandardScaler for numerical features

The app is deployed with **Streamlit** for easy user interaction.

---

## ⚙️ Features
- Predict customer churn using simple inputs.
- Handles categorical data like Country and Gender automatically.
- Displays churn probability.
- Clean Streamlit interface.

---

## 🧠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 🧩 Files in the Project
| File | Description |
|------|--------------|
| `train_model.py` | Script to train and save the model pipeline |
| `churn_model.pkl` | Trained ML pipeline including preprocessing |
| `app.py` | Streamlit web app for prediction |
| `requirements.txt` | Dependencies list |
| `Bank Customer Churn Prediction.csv` | Dataset |

---

## 🚀 Run Locally
Clone the repository and run:

```bash
git clone https://github.com/<your-username>/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
