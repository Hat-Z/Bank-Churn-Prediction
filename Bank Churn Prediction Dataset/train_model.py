import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
# Load dataset
df = pd.read_csv("Bank Customer Churn Prediction.csv")
# Drop customer_id
df.drop("customer_id", axis=1, inplace=True)
# Split into X and y
X = df.drop("churn", axis=1)
y = df["churn"]
# Define categorical and numeric columns
categorical_features = ["country", "gender"]
numeric_features = ["credit_score", "age", "tenure", "balance", 
                    "products_number", "credit_card", "active_member", "estimated_salary"]

# Preprocessing steps
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numeric_transformer, numeric_features)
    ]
)
# Combine preprocessing and model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model.fit(X_train, y_train)
# Save trained pipeline
joblib.dump(model, "churn_model.pkl")
print("✅ Model trained successfully and saved as churn_model.pkl (with preprocessing included).")
