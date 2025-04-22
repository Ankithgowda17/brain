import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Page config
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="centered")

# Load model
with open("stroke_model.pkl", "rb") as file:
    model = pickle.load(file)

# App header
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #4B8BBE;'>🧠 Stroke Risk Prediction</h1>
        <p style='font-size:18px;'>Predict the likelihood of stroke based on health inputs.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Input form layout
with st.form("prediction_form"):
    st.subheader("🔍 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", min_value=1, max_value=120, value=30)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])

    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, value=22.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submit_button = st.form_submit_button(label="🚀 Predict Stroke Risk")

# Map and prepare data
def prepare_input():
    return pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }])

# Prediction logic
if submit_button:
    input_df = prepare_input()
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("🧾 Prediction Result:")

    if prediction == 1:
        st.error("⚠️ The model predicts a **HIGH RISK** of stroke.")
        st.markdown("> Please consult a healthcare provider for further evaluation.")
    else:
        st.success("✅ The model predicts a **LOW RISK** of stroke.")
        st.markdown("> Keep maintaining a healthy lifestyle!")

# Footer
st.markdown("---")

