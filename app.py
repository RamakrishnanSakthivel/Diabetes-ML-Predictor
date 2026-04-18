from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Diabetes Prediction API", version="1.0")

# Load model and scaler once at startup
model = joblib.load("./local_model/model.pkl")
scaler = joblib.load("./local_model/scaler.pkl")

# Define the input schema
class PatientData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float

class PredictionResult(BaseModel):
    prediction: str
    is_diabetic: bool
    confidence: float

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running! ✅"}

@app.post("/predict", response_model=PredictionResult)
def predict(patient: PatientData):
    # Prepare input
    data = np.array([[
        patient.pregnancies,
        patient.glucose,
        patient.blood_pressure,
        patient.skin_thickness,
        patient.insulin,
        patient.bmi,
        patient.diabetes_pedigree,
        patient.age
    ]])

    # Scale and predict
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0]

    return PredictionResult(
        prediction="Diabetic" if prediction == 1 else "Not Diabetic",
        is_diabetic=bool(prediction),
        confidence=round(float(max(probability)) * 100, 2)
    )

@app.get("/health")
def health():
    return {"status": "healthy"}