import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load trained model
with open('gradient_boosting_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load target encoder for smoking_status
with open('target_encoder.pkl', 'rb') as te_file:
    target_encoder = pickle.load(te_file)

# Label Encoding Function
def label_encode(value, categories):
    encoder = LabelEncoder()
    encoder.fit(categories)
    return encoder.transform([value])[0]

# Streamlit UI
st.title("Stroke Prediction App")
st.write("Enter the details below to predict stroke risk.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
age = st.number_input("Age", min_value=0, max_value=100, value=50)
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

# Encode categorical inputs
encoded_gender = label_encode(gender, ["Male", "Female", "Other"])
encoded_ever_married = label_encode(ever_married, ["Yes", "No"])
encoded_work_type = label_encode(work_type, ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
encoded_residence_type = label_encode(residence_type, ["Urban", "Rural"])

smoking_df = pd.DataFrame({"smoking_status": [smoking_status]})
encoded_smoking_status = target_encoder.transform(smoking_df)["smoking_status"].values[0]

# Convert binary features
dict_binary = {"No": 0, "Yes": 1}
encoded_hypertension = dict_binary[hypertension]
encoded_heart_disease = dict_binary[heart_disease]

# Create dataframe for prediction
input_data = np.array([[
    encoded_gender, encoded_ever_married, encoded_work_type, encoded_residence_type,
    encoded_smoking_status, age, avg_glucose_level, bmi, encoded_hypertension, encoded_heart_disease
]])

test_df = pd.DataFrame(input_data, columns=[
    "gender", "ever_married", "work_type", "Residence_type", "smoking_status", 
    "age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"
])

# Predict button
if st.button("Predict Stroke Risk"):
    prediction = model.predict(test_df)
    probability = model.predict_proba(test_df)[0][1] * 100
    
    if prediction[0] == 1:
        st.error(f"High Risk: {probability:.2f}% chance of stroke")
    else:
        st.success(f"Low Risk: {probability:.2f}% chance of stroke")

st.write("\n**Disclaimer:** This prediction is for informational purposes only and not medical advice.")
