import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Function to reorder DataFrame columns
def order_data(df):
    desired_order = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    # Reorder the DataFrame columns
    decoded = df[desired_order]
    return decoded

# Function to encode the raw input data
def custom_encode(raw_input):
    # Encoding for 'gender'
    if raw_input['gender'] == 'Female':
        raw_input['gender'] = 0
    elif raw_input['gender'] == 'Male':
        raw_input['gender'] = 1
    else:
        raise ValueError("Invalid gender value")

    # Encoding for 'ever_married'
    if raw_input['ever_married'] == 'No':
        raw_input['ever_married'] = 0
    elif raw_input['ever_married'] == 'Yes':
        raw_input['ever_married'] = 1
    else:
        raise ValueError("Invalid ever_married value")

    # Encoding for 'work_type'
    if raw_input['work_type'] == 'Govt_job':
        raw_input['work_type'] = 0
    elif raw_input['work_type'] == 'Private':
        raw_input['work_type'] = 1
    elif raw_input['work_type'] == 'Self-employed':
        raw_input['work_type'] = 2
    elif raw_input['work_type'] == 'children':
        raw_input['work_type'] = 3
    else:
        raise ValueError("Invalid work_type value")

    # Encoding for 'Residence_type'
    if raw_input['Residence_type'] == 'Rural':
        raw_input['Residence_type'] = 0
    elif raw_input['Residence_type'] == 'Urban':
        raw_input['Residence_type'] = 1
    else:
        raise ValueError("Invalid Residence_type value")

    # Encoding for 'smoking_status'
    if raw_input['smoking_status'] == 'Unknown':
        raw_input['smoking_status'] = 0
    elif raw_input['smoking_status'] == 'formerly smoked':
        raw_input['smoking_status'] = 1
    elif raw_input['smoking_status'] == 'never smoked':
        raw_input['smoking_status'] = 2
    elif raw_input['smoking_status'] == 'smokes':
        raw_input['smoking_status'] = 3
    else:
        raise ValueError("Invalid smoking_status value")

    # Encoding for 'hypertension' and 'heart_disease' (same as before)
    raw_input['hypertension'] = 1 if raw_input['hypertension'] == "Yes" else 0
    raw_input['heart_disease'] = 1 if raw_input['heart_disease'] == "Yes" else 0

    return raw_input

# Page config
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="centered")

# Load model
with open("rfcl.pickle", "rb") as file:
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
    # Prepare the data from form inputs
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }])

    # Encode the raw input data
    input_data = custom_encode(input_data.iloc[0].to_dict())

    # Convert back to DataFrame
    input_data_df = pd.DataFrame([input_data])

    # Reorder columns to match model's expected input
    input_data_df = order_data(input_data_df)
    return input_data_df

# Prediction logic
if submit_button:
    input_df = prepare_input()
    
    # Make the prediction
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
