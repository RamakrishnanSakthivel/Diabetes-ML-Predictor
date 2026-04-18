# predict.py - to test the model locally before deploying
import joblib
import numpy as np

# Load the model files (download from Azure first)
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="ca7f03ae-3bb4-43fd-8599-d4b1fbafe848",
    resource_group_name="diabetes-rg",
    workspace_name="diabetes-ml-workspace"
)

# Download the registered model
ml_client.models.download(
    name="diabetes-predictor",
    version="1",
    download_path="./downloaded_model"
)
print("Model downloaded! ✅")

# Load model and scaler
model = joblib.load("./downloaded_model/model_output/model.pkl")
scaler = joblib.load("./downloaded_model/model_output/scaler.pkl")

# Test prediction — [Pregnancies, Glucose, BloodPressure, SkinThickness,
#                     Insulin, BMI, DiabetesPedigreeFunction, Age]
test_patient = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

test_scaled = scaler.transform(test_patient)
prediction = model.predict(test_scaled)
probability = model.predict_proba(test_scaled)

print("\n--- Diabetes Prediction Result ---")
print(f"Patient data: Glucose=148, BMI=33.6, Age=50")
print(f"Prediction: {'🔴 Diabetic' if prediction[0] == 1 else '🟢 Not Diabetic'}")
print(f"Confidence: {max(probability[0]) * 100:.1f}%")