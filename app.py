# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model, scaler, and column names
model = joblib.load('Logistic_regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# Title
st.title("💓 Heart Disease Prediction App")

# Sidebar inputs
st.sidebar.header("Patient Data")

age = st.sidebar.number_input("Age", 1, 120, 50)
resting_bp = st.sidebar.number_input("RestingBP", 0, 200, 120)
cholesterol = st.sidebar.number_input("Cholesterol", 0, 600, 200)
max_hr = st.sidebar.number_input("MaxHR", 0, 250, 150)
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 1.0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["N", "Y"])
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prepare input dict
input_data = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_M': 1 if sex == 'Male' else 0,
    'ChestPainType_ASY': int(chest_pain == 'ASY'),
    'ChestPainType_NAP': int(chest_pain == 'NAP'),
    'ChestPainType_TA': int(chest_pain == 'TA'),
    'FastingBS': fasting_bs,
    'RestingECG_Normal': int(resting_ecg == 'Normal'),
    'RestingECG_ST': int(resting_ecg == 'ST'),
    'ExerciseAngina_Y': int(exercise_angina == 'Y'),
    'ST_Slope_Flat': int(st_slope == 'Flat'),
    'ST_Slope_Up': int(st_slope == 'Up')
}

df = pd.DataFrame([input_data])

# Scale numerical columns
df_scaled = scaler.transform(df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])
df[df.columns[:5]] = df_scaled

# Reindex to match training columns
df = df.reindex(columns=columns, fill_value=0)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(df)[0]
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ High chance of Heart Disease.")
    else:
        st.success("✅ No Heart Disease detected.")
