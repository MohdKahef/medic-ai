"""
Train all three disease prediction models using synthetic datasets.
Run this script once to generate .pkl model files.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
N = 1000

os.makedirs("../models", exist_ok=True)
os.makedirs("../Data", exist_ok=True)

# ─────────────────────────────────────────────
# 1. HEART DISEASE DATASET
# ─────────────────────────────────────────────
# Features: age, sex, cp, trestbps, chol, fbs, restecg,
#           thalach, exang, oldpeak, slope, ca, thal
heart_data = {
    "age":      np.random.randint(29, 78, N),
    "sex":      np.random.randint(0, 2, N),
    "cp":       np.random.randint(0, 4, N),
    "trestbps": np.random.randint(94, 200, N),
    "chol":     np.random.randint(126, 564, N),
    "fbs":      np.random.randint(0, 2, N),
    "restecg":  np.random.randint(0, 3, N),
    "thalach":  np.random.randint(71, 202, N),
    "exang":    np.random.randint(0, 2, N),
    "oldpeak":  np.round(np.random.uniform(0, 6.2, N), 1),
    "slope":    np.random.randint(0, 3, N),
    "ca":       np.random.randint(0, 4, N),
    "thal":     np.random.randint(0, 4, N),
}
df_heart = pd.DataFrame(heart_data)
# Synthetic target: high risk if older, high chol, low thalach
risk_score = (
    (df_heart["age"] > 55).astype(int) +
    (df_heart["chol"] > 240).astype(int) +
    (df_heart["thalach"] < 140).astype(int) +
    (df_heart["trestbps"] > 140).astype(int) +
    (df_heart["ca"] > 0).astype(int)
)
df_heart["target"] = (risk_score >= 3).astype(int)
df_heart.to_csv("../Data/heart.csv", index=False)

X_h = df_heart.drop("target", axis=1)
y_h = df_heart["target"]
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(X_h, y_h, test_size=0.2)
scaler_h = StandardScaler()
X_h_train_s = scaler_h.fit_transform(X_h_train)
X_h_test_s  = scaler_h.transform(X_h_test)
model_h = RandomForestClassifier(n_estimators=100, random_state=42)
model_h.fit(X_h_train_s, y_h_train)
acc_h = accuracy_score(y_h_test, model_h.predict(X_h_test_s))
joblib.dump(model_h,  "../models/heart_model.pkl")
joblib.dump(scaler_h, "../models/heart_scaler.pkl")
print(f"Heart Disease Model  — Accuracy: {acc_h:.2%}")

# ─────────────────────────────────────────────
# 2. LIVER DISEASE DATASET (Indian Liver Patient)
# ─────────────────────────────────────────────
# Features: Age, Gender, Total_Bilirubin, Direct_Bilirubin,
#           Alkaline_Phosphotase, Alamine_Aminotransferase,
#           Aspartate_Aminotransferase, Total_Protiens,
#           Albumin, Albumin_and_Globulin_Ratio
liver_data = {
    "Age":                          np.random.randint(4, 90, N),
    "Gender":                       np.random.randint(0, 2, N),
    "Total_Bilirubin":              np.round(np.random.uniform(0.4, 75, N), 1),
    "Direct_Bilirubin":             np.round(np.random.uniform(0.1, 19, N), 1),
    "Alkaline_Phosphotase":         np.random.randint(63, 2110, N),
    "Alamine_Aminotransferase":     np.random.randint(10, 2000, N),
    "Aspartate_Aminotransferase":   np.random.randint(10, 4929, N),
    "Total_Protiens":               np.round(np.random.uniform(2.7, 9.6, N), 1),
    "Albumin":                      np.round(np.random.uniform(0.9, 5.5, N), 1),
    "Albumin_and_Globulin_Ratio":   np.round(np.random.uniform(0.3, 2.8, N), 2),
}
df_liver = pd.DataFrame(liver_data)
risk_score_l = (
    (df_liver["Total_Bilirubin"] > 2).astype(int) +
    (df_liver["Alkaline_Phosphotase"] > 200).astype(int) +
    (df_liver["Alamine_Aminotransferase"] > 56).astype(int) +
    (df_liver["Aspartate_Aminotransferase"] > 40).astype(int) +
    (df_liver["Albumin"] < 3.5).astype(int)
)
df_liver["Dataset"] = (risk_score_l >= 3).astype(int)
df_liver.to_csv("../Data/liver.csv", index=False)

X_l = df_liver.drop("Dataset", axis=1)
y_l = df_liver["Dataset"]
X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(X_l, y_l, test_size=0.2)
scaler_l = StandardScaler()
X_l_train_s = scaler_l.fit_transform(X_l_train)
X_l_test_s  = scaler_l.transform(X_l_test)
model_l = RandomForestClassifier(n_estimators=100, random_state=42)
model_l.fit(X_l_train_s, y_l_train)
acc_l = accuracy_score(y_l_test, model_l.predict(X_l_test_s))
joblib.dump(model_l,  "../models/liver_model.pkl")
joblib.dump(scaler_l, "../models/liver_scaler.pkl")
print(f"Liver Disease Model  — Accuracy: {acc_l:.2%}")

# ─────────────────────────────────────────────
# 3. DIABETES DATASET (Pima Indians style)
# ─────────────────────────────────────────────
# Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
#           Insulin, BMI, DiabetesPedigreeFunction, Age
diabetes_data = {
    "Pregnancies":              np.random.randint(0, 17, N),
    "Glucose":                  np.random.randint(44, 199, N),
    "BloodPressure":            np.random.randint(24, 122, N),
    "SkinThickness":            np.random.randint(7, 99, N),
    "Insulin":                  np.random.randint(14, 846, N),
    "BMI":                      np.round(np.random.uniform(18.2, 67.1, N), 1),
    "DiabetesPedigreeFunction": np.round(np.random.uniform(0.078, 2.42, N), 3),
    "Age":                      np.random.randint(21, 81, N),
}
df_diabetes = pd.DataFrame(diabetes_data)
risk_score_d = (
    (df_diabetes["Glucose"] > 140).astype(int) +
    (df_diabetes["BMI"] > 30).astype(int) +
    (df_diabetes["Age"] > 45).astype(int) +
    (df_diabetes["DiabetesPedigreeFunction"] > 0.5).astype(int) +
    (df_diabetes["Pregnancies"] > 5).astype(int)
)
df_diabetes["Outcome"] = (risk_score_d >= 3).astype(int)
df_diabetes.to_csv("../Data/diabetes.csv", index=False)

X_d = df_diabetes.drop("Outcome", axis=1)
y_d = df_diabetes["Outcome"]
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(X_d, y_d, test_size=0.2)
scaler_d = StandardScaler()
X_d_train_s = scaler_d.fit_transform(X_d_train)
X_d_test_s  = scaler_d.transform(X_d_test)
model_d = RandomForestClassifier(n_estimators=100, random_state=42)
model_d.fit(X_d_train_s, y_d_train)
acc_d = accuracy_score(y_d_test, model_d.predict(X_d_test_s))
joblib.dump(model_d,  "../models/diabetes_model.pkl")
joblib.dump(scaler_d, "../models/diabetes_scaler.pkl")
print(f"Diabetes Model       — Accuracy: {acc_d:.2%}")

print("\n✅ All models trained and saved successfully!")
print("   models/heart_model.pkl, heart_scaler.pkl")
print("   models/liver_model.pkl, liver_scaler.pkl")
print("   models/diabetes_model.pkl, diabetes_scaler.pkl")
