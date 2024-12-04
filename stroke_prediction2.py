import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import joblib

# Load the trained model (from the pipeline)
model = joblib.load('ada_boost_model.pkl')  # Load your trained model
scaler = joblib.load('scaler.pkl') #load scaler

# Feature Engineering Function
def feature_engineering(df):
    # Calculate risk factor based on the conditions provided
    df['risk_factor'] = df[['age','avg_glucose_level','bmi',
                             'heart_disease', 'hypertension', 'smoking_status']].apply(
        lambda x: 0 + (1 if x.age >= 45 else 0) + 
                  (1 if x.bmi > 24.99 else 0) + 
                  (1 if x.avg_glucose_level > 99 else 0) + 
                  (1 if x.heart_disease == 1 else 0) + 
                  (1 if x.hypertension == 1 else 0) + 
                  (1 if x.smoking_status in ['formerly smoked', 'smokes'] else 0), axis=1)

    # Return the dataframe with all the new engineered features
    return df

# Streamlit input interface
st.title('Stroke Prediction')

# User input fields
age = st.number_input('Age', min_value=1, max_value=100)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=50.0, max_value=300.0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0)
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])
gender = st.selectbox('Gender', ['Male', 'Female'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])

# When user presses the button to predict
if st.button('Predict Stroke Risk'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'age': [age],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'smoking_status': [smoking_status],
        'gender': [gender],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'ever_married': [ever_married]
    })
    
    nominal_features = ['gender', 'work_type', 'Residence_type', 'ever_married']
    # Apply OneHotEncoder with pre-defined categories to ensure consistency
    categories = [['Female', 'Male'], 
                  ['Govt_job', 'Private', 'Self-employed', 'children', 'Never_worked'], 
                  ['Rural', 'Urban'], 
                  ['No', 'Yes']]
    one_hot_encoder = OneHotEncoder(categories=categories, sparse_output=False, drop='first')

    # Apply feature engineering
    input_data = feature_engineering(input_data)
    
    # Create an ordinal encoder for smoking status
    ordinal_encoder = OrdinalEncoder(categories=[['Unknown', 'never smoked', 'formerly smoked', 'smokes']])
    input_data['smoking_status'] = ordinal_encoder.fit_transform(input_data[['smoking_status']])
    
    # Display the input data after ordinal encoding for debugging
    #st.write("Input Data after Ordinal Encoding:")
    #st.dataframe(input_data)

    # OneHotEncoder for categorical features (gender, work_type, Residence_type, ever_married)
    one_hot_encoder = OneHotEncoder(categories=categories, sparse_output=False, drop='first')

    # Apply OneHotEncoder
    one_hot_encoded = one_hot_encoder.fit_transform(input_data[nominal_features])

    # Convert to DataFrame
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(nominal_features))

    # Ensure indices are aligned
    one_hot_encoded_df.reset_index(drop=True, inplace=True)
    input_data.reset_index(drop=True, inplace=True)

    # Concatenate the original input_data without the nominal columns with the one-hot encoded columns
    input_data = pd.concat([input_data.drop(nominal_features, axis=1), one_hot_encoded_df], axis=1)
    
    # Display the final input_data with one-hot encoded columns
    #st.write("Input Data after One-Hot Encoding:")
    #st.dataframe(one_hot_encoded_df)

    # Standard scaling for numeric features like age, bmi, avg_glucose_level
    # Apply the scaler to the input data
    input_data_scaled = scaler.transform(input_data[['age', 'bmi', 'avg_glucose_level']])
    
    
    # Ensure that the final input features match the shape that the model expects (14 features)
    # After scaling, ensure that you include the missing features from the model (heart_disease, hypertension, etc.)
    input_data_final_scaled = pd.DataFrame(input_data_scaled, columns=['age', 'avg_glucose_level', 'bmi'])
    
    input_data[['age', 'bmi', 'avg_glucose_level']] = input_data_final_scaled
    
    # Display the final input_data with one-hot encoded columns
    st.write("Input Data after Scaling:")
    st.dataframe(input_data)

    expected_feature_order = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 
    'smoking_status', 'gender_Male', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 
    'work_type_children', 'Residence_type_Urban', 
    'ever_married_Yes', 'risk_factor']
    
    input_data = input_data[expected_feature_order]

    
     # Display the final input_data with one-hot encoded columns
    st.write("Input Data after Preprocessing:")
    st.dataframe(input_data)

    # Now use the trained model to make a prediction
    try:
        # Use the model to make predictions
        prediction = model.predict(input_data)

        # Get predicted probabilities
        prediction_prob = model.predict_proba(input_data)

        # Display the results
        st.write(f"Probability of No Stroke (0): {prediction_prob[0][0]:.4f}")
        st.write(f"Probability of Stroke (1): {prediction_prob[0][1]:.4f}")

        # Decision based on the threshold (e.g., 0.5)
        if prediction_prob[0][1] > 0.5:
            st.write("The model predicts a high risk of stroke.")
        else:
            st.write("The model predicts a low risk of stroke.")
        
    except Exception as e:
        st.write(f"Error in prediction: {e}")
