# train.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Path to data")
parser.add_argument("--model_output", type=str, help="Path to save model")
args = parser.parse_args()

# Load the Pima Indians Diabetes dataset (from sklearn or URL)
from sklearn.datasets import load_diabetes
from urllib.request import urlretrieve

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]

urlretrieve(url, "diabetes.csv")
df = pd.read_csv("diabetes.csv", names=cols)

# Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(classification_report(y_test, preds))

# Save model
os.makedirs(args.model_output, exist_ok=True)
joblib.dump(model, os.path.join(args.model_output, "model.pkl"))
joblib.dump(scaler, os.path.join(args.model_output, "scaler.pkl"))
print("Model saved!")  