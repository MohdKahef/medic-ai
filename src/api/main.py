"""
Personalized Medicine Prediction System — FastAPI Backend
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

app = FastAPI(
    title="Personalized Medicine Prediction API",
    description="AI-powered disease prediction for Heart, Liver, and Diabetes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model paths (relative to this file) ──────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..", "..", "models")

def load(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

heart_model    = load("heart_model.pkl")
heart_scaler   = load("heart_scaler.pkl")
liver_model    = load("liver_model.pkl")
liver_scaler   = load("liver_scaler.pkl")
diabetes_model = load("diabetes_model.pkl")
diabetes_scaler= load("diabetes_scaler.pkl")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class HeartInput(BaseModel):
    age: float
    sex: float          # 0=Female, 1=Male
    cp: float           # Chest pain type 0-3
    trestbps: float     # Resting blood pressure
    chol: float         # Serum cholesterol mg/dl
    fbs: float          # Fasting blood sugar > 120 mg/dl (1=True)
    restecg: float      # Resting ECG results 0-2
    thalach: float      # Maximum heart rate achieved
    exang: float        # Exercise induced angina (1=Yes)
    oldpeak: float      # ST depression induced by exercise
    slope: float        # Slope of peak exercise ST segment 0-2
    ca: float           # Number of major vessels 0-3
    thal: float         # Thal: 0=Normal, 1=Fixed defect, 2=Reversable defect

class LiverInput(BaseModel):
    age: float
    gender: float       # 0=Female, 1=Male
    total_bilirubin: float
    direct_bilirubin: float
    alkaline_phosphotase: float
    alamine_aminotransferase: float
    aspartate_aminotransferase: float
    total_protiens: float
    albumin: float
    albumin_and_globulin_ratio: float

class DiabetesInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float


# ── Response builder ──────────────────────────────────────────────────────────

def build_response(disease: str, prediction: int, probability: float):
    risk_level = "Low" if probability < 0.35 else "Moderate" if probability < 0.65 else "High"
    risk_color = {"Low": "green", "Moderate": "orange", "High": "red"}[risk_level]

    responses = {
        "heart": {
            "positive": {
                "lifestyle": [
                    "Follow a heart-healthy diet low in saturated fat and sodium",
                    "Exercise at least 30 minutes of moderate activity daily",
                    "Quit smoking and limit alcohol consumption",
                    "Monitor blood pressure and cholesterol regularly",
                    "Manage stress through yoga or meditation",
                ],
                "medications": [
                    "Statins (e.g., Atorvastatin) — for cholesterol management",
                    "ACE Inhibitors (e.g., Lisinopril) — for blood pressure",
                    "Beta-blockers (e.g., Metoprolol) — for heart rate control",
                    "Aspirin (low dose) — antiplatelet therapy",
                ],
                "diet": [
                    "Mediterranean diet: olive oil, fish, vegetables, whole grains",
                    "Reduce red meat; increase omega-3 rich foods",
                    "Limit sodium to < 1500 mg/day",
                    "Increase potassium: bananas, sweet potatoes, spinach",
                ],
                "consult": True,
            },
            "negative": {
                "lifestyle": [
                    "Maintain current healthy lifestyle habits",
                    "Regular annual cardiovascular checkups",
                    "Stay physically active — 150 min/week moderate exercise",
                    "Maintain healthy weight (BMI 18.5–24.9)",
                ],
                "medications": ["No medication currently required", "Preventive low-dose aspirin — consult your doctor"],
                "diet": ["Continue balanced diet rich in fruits and vegetables", "Stay hydrated; limit processed foods"],
                "consult": False,
            },
        },
        "liver": {
            "positive": {
                "lifestyle": [
                    "Abstain from alcohol completely",
                    "Avoid hepatotoxic medications (NSAIDs, acetaminophen excess)",
                    "Maintain healthy weight to prevent fatty liver progression",
                    "Get vaccinated for Hepatitis A and B if not done",
                ],
                "medications": [
                    "Ursodeoxycholic acid — for bile acid regulation",
                    "Silymarin (Milk Thistle) — hepatoprotective supplement",
                    "Vitamin E — antioxidant for non-alcoholic fatty liver",
                    "Lactulose — if hepatic encephalopathy suspected",
                ],
                "diet": [
                    "High-fiber diet: oats, legumes, fruits",
                    "Avoid fatty, fried, processed foods",
                    "Increase antioxidants: broccoli, berries, green tea",
                    "Limit fructose and refined sugars",
                ],
                "consult": True,
            },
            "negative": {
                "lifestyle": [
                    "Moderate alcohol (or none) is best for liver health",
                    "Annual liver function tests recommended",
                    "Stay at a healthy weight",
                ],
                "medications": ["No medication required", "Liver-supportive supplements optional — discuss with physician"],
                "diet": ["Balanced nutrition; limit alcohol", "Coffee consumption may be protective"],
                "consult": False,
            },
        },
        "diabetes": {
            "positive": {
                "lifestyle": [
                    "Monitor blood glucose daily — target fasting < 100 mg/dL",
                    "Exercise: 150 min/week aerobic + resistance training",
                    "Lose 5–10% body weight if overweight",
                    "Quit smoking; alcohol only in moderation",
                    "Regular foot and eye examinations",
                ],
                "medications": [
                    "Metformin — first-line oral hypoglycemic",
                    "SGLT-2 inhibitors (e.g., Empagliflozin) — glucose excretion",
                    "GLP-1 agonists (e.g., Semaglutide) — weight + sugar control",
                    "Insulin therapy — if oral agents insufficient",
                ],
                "diet": [
                    "Low-glycemic index foods: whole grains, legumes, non-starchy vegetables",
                    "Limit simple sugars and refined carbs",
                    "Portion control; eat smaller meals more frequently",
                    "Increase dietary fiber to 25–30 g/day",
                ],
                "consult": True,
            },
            "negative": {
                "lifestyle": [
                    "Maintain current activity and diet patterns",
                    "Annual HbA1c and fasting glucose screening",
                    "Reduce sugar-sweetened beverage intake",
                ],
                "medications": ["No medication required at this time", "Consider metformin if pre-diabetic — consult doctor"],
                "diet": ["Continue a balanced, whole-food diet", "Limit processed sugars and white flour"],
                "consult": False,
            },
        },
    }

    key = "positive" if prediction == 1 else "negative"
    info = responses[disease][key]

    return {
        "disease": disease.capitalize() + " Disease" if disease != "diabetes" else "Diabetes",
        "prediction": int(prediction),
        "prediction_label": "High Risk — Disease Detected" if prediction == 1 else "Low Risk — No Disease Detected",
        "probability": round(float(probability), 4),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "lifestyle_suggestions": info["lifestyle"],
        "medication_suggestions": info["medications"],
        "diet_recommendations": info["diet"],
        "doctor_consultation_recommended": info["consult"],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Personalized Medicine Prediction API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": ["heart", "liver", "diabetes"]}


@app.post("/predict-heart")
def predict_heart(data: HeartInput):
    try:
        features = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]])
        scaled = heart_scaler.transform(features)
        pred   = heart_model.predict(scaled)[0]
        prob   = heart_model.predict_proba(scaled)[0][1]
        return build_response("heart", pred, prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-liver")
def predict_liver(data: LiverInput):
    try:
        features = np.array([[
            data.age, data.gender, data.total_bilirubin, data.direct_bilirubin,
            data.alkaline_phosphotase, data.alamine_aminotransferase,
            data.aspartate_aminotransferase, data.total_protiens,
            data.albumin, data.albumin_and_globulin_ratio
        ]])
        scaled = liver_scaler.transform(features)
        pred   = liver_model.predict(scaled)[0]
        prob   = liver_model.predict_proba(scaled)[0][1]
        return build_response("liver", pred, prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        features = np.array([[
            data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree_function, data.age
        ]])
        scaled = diabetes_scaler.transform(features)
        pred   = diabetes_model.predict(scaled)[0]
        prob   = diabetes_model.predict_proba(scaled)[0][1]
        return build_response("diabetes", pred, prob)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
