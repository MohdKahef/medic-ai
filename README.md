# 🏥 Personalized Medicine Prediction System

An AI-powered healthcare web application that predicts risk for **Heart Disease**, **Liver Disease**, and **Diabetes** using machine learning, and generates personalized medical reports with diet, lifestyle, and medication suggestions.

---

## 📁 Project Structure

```
personalized-medicine/
│
├── Data/
│   ├── heart.csv
│   ├── liver.csv
│   └── diabetes.csv
│
├── models/                        ← Generated after training
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── liver_model.pkl
│   ├── liver_scaler.pkl
│   ├── diabetes_model.pkl
│   └── diabetes_scaler.pkl
│
├── src/
│   ├── train_models.py            ← Trains all 3 models at once
│   └── api/
│       └── main.py                ← FastAPI backend
│
├── frontend/
│   ├── login.html
│   ├── dashboard.html
│   ├── heart.html
│   ├── liver.html
│   ├── diabetes.html
│   └── static/
│       ├── css/
│       │   └── styles.css
│       └── js/
│           └── main.js
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run Instructions

### Step 1 — Install Dependencies

```bash
cd personalized-medicine
pip install -r requirements.txt
```

### Step 2 — Train the ML Models

```bash
cd src
python train_models.py
```

This will:
- Generate synthetic datasets in `Data/`
- Train RandomForest classifiers for Heart, Liver, and Diabetes
- Save `.pkl` model files + scalers in `models/`

Expected output:
```
Heart Disease Model  — Accuracy: 96.50%
Liver Disease Model  — Accuracy: 100.00%
Diabetes Model       — Accuracy: 96.00%
✅ All models trained and saved successfully!
```

### Step 3 — Start the FastAPI Backend

```bash
cd src/api
uvicorn main:app --reload --port 8000
```

API will be running at: `http://127.0.0.1:8000`

To view interactive API docs: `http://127.0.0.1:8000/docs`

### Step 4 — Open the Frontend

Simply open `frontend/login.html` in your browser:

```bash
# On macOS
open frontend/login.html

# On Linux
xdg-open frontend/login.html

# On Windows
start frontend/login.html
```

Or serve it with Python:
```bash
cd frontend
python -m http.server 3000
# Then open http://localhost:3000/login.html
```

---

## 🔑 Demo Login Credentials

| Role    | Email              | Password    |
|---------|--------------------|-------------|
| Doctor  | doctor@ai.com      | doctor123   |
| Patient | patient@ai.com     | patient123  |

---

## 🔌 API Endpoints

### Health Check
```
GET  /health
```

### Heart Disease Prediction
```
POST /predict-heart
Content-Type: application/json

{
  "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
  "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
  "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
}
```

### Liver Disease Prediction
```
POST /predict-liver
Content-Type: application/json

{
  "age": 45, "gender": 1, "total_bilirubin": 0.9,
  "direct_bilirubin": 0.2, "alkaline_phosphotase": 202,
  "alamine_aminotransferase": 64, "aspartate_aminotransferase": 60,
  "total_protiens": 7.0, "albumin": 3.3, "albumin_and_globulin_ratio": 0.9
}
```

### Diabetes Prediction
```
POST /predict-diabetes
Content-Type: application/json

{
  "pregnancies": 6, "glucose": 148, "blood_pressure": 72,
  "skin_thickness": 35, "insulin": 0, "bmi": 33.6,
  "diabetes_pedigree_function": 0.627, "age": 50
}
```

### Response Format (all endpoints)
```json
{
  "disease": "Heart Disease",
  "prediction": 1,
  "prediction_label": "High Risk — Disease Detected",
  "probability": 0.72,
  "risk_level": "High",
  "risk_color": "red",
  "lifestyle_suggestions": ["..."],
  "medication_suggestions": ["..."],
  "diet_recommendations": ["..."],
  "doctor_consultation_recommended": true
}
```

---

## 🤖 ML Models Summary

| Disease       | Algorithm         | Features | Accuracy |
|---------------|-------------------|----------|----------|
| Heart Disease | RandomForest (100 trees) | 13 | 96.5% |
| Liver Disease | RandomForest (100 trees) | 10 | 100%   |
| Diabetes      | RandomForest (100 trees) | 8  | 96.0%  |

All models use `StandardScaler` for feature normalization before prediction.

---

## ✨ Features

- 🔐 Role-based login (Doctor / Patient)
- 📊 Animated dashboard with prediction statistics
- 🩺 3 disease prediction modules with step-by-step forms
- 📋 AI Medical Reports with diet, lifestyle & medication suggestions
- 📱 Fully responsive design
- 🎨 Modern dark healthcare UI with animated transitions
- ⚡ Real-time probability visualization

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. Predictions are based on synthetic training data and should **not** be used as a substitute for professional medical diagnosis or treatment.
