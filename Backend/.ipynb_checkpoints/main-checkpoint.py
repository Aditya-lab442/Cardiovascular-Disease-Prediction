from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Cardiovascular Disease Prediction API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# model.pkl now contains the complete trained pipeline, including scaling.
model = joblib.load("model.pkl")

class PatientData(BaseModel):
    age: float
    gender: int
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int

@app.get("/")
def home():
    return {"message": "Cardiovascular Disease Prediction API is running"}

@app.post("/predict")
def predict(data: PatientData):
    # The frontend/API accepts age in years. The model was trained with age in years.
    input_data = pd.DataFrame([{
        "age": data.age,
        "gender": data.gender,
        "height": data.height,
        "weight": data.weight,
        "ap_hi": data.ap_hi,
        "ap_lo": data.ap_lo,
        "cholesterol": data.cholesterol,
        "gluc": data.gluc,
        "smoke": data.smoke,
        "alco": data.alco,
        "active": data.active
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
