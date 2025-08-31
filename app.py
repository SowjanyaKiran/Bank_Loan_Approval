# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ========== 1. Load Pickle Model ==========
# Ensure the model file is accessible in the same directory
try:
    with open("loan_approval_model.pkl", "rb") as f:
        model, scaler, le = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'loan_approval_model.pkl' not found. Please run the 'train_model.py' script first to create the model file.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model file: {e}")
    st.stop()


st.title("🏦 Loan Approval Prediction App")
st.markdown("Fill the applicant details and check loan approval status.")

# ========== 2. User Inputs ==========
st.subheader("Applicant Information")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0.0, step=100.0)
coapplicant_income = st.number_input("Co-applicant Income", min_value=0.0, step=100.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, step=10.0)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0.0, step=12.0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.button("Predict Loan Approval"):
    # Create a dictionary to hold the raw input data
    input_dict = {
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Gender_Female': [1 if gender == 'Female' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'Married_No': [1 if married == 'No' else 0],
        'Married_Yes': [1 if married == 'Yes' else 0],
        'Dependents_0': [1 if dependents == '0' else 0],
        'Dependents_1': [1 if dependents == '1' else 0],
        'Dependents_2': [1 if dependents == '2' else 0],
        'Dependents_3+': [1 if dependents == '3+' else 0],
        'Education_Graduate': [1 if education == 'Graduate' else 0],
        'Education_Not Graduate': [1 if education == 'Not Graduate' else 0],
        'Self_Employed_No': [1 if self_employed == 'No' else 0],
        'Self_Employed_Yes': [1 if self_employed == 'Yes' else 0],
        'Property_Area_Rural': [1 if property_area == 'Rural' else 0],
        'Property_Area_Semiurban': [1 if property_area == 'Semiurban' else 0],
        'Property_Area_Urban': [1 if property_area == 'Urban' else 0]
    }
    
    # Convert the dictionary to a pandas DataFrame
    # The order of columns is crucial and must match the order from training
    input_df = pd.DataFrame(input_dict)
    
    # Scale the input data using the loaded scaler
    try:
        X_input_scaled = scaler.transform(input_df)
    except ValueError as e:
        st.error(f"ValueError during scaling. Please ensure all input fields are filled correctly and the number of features matches the trained scaler. Error details: {e}")
        st.stop()

    # Make the prediction
    prediction = model.predict(X_input_scaled)[0]
    
    # Get the probabilities for each class
    prediction_proba = model.predict_proba(X_input_scaled)[0]
    
    # Note: `le.classes_` will give the order of classes (e.g., ['N', 'Y'])
    # So we can use the label encoder to map probabilities correctly.
    prob_rejection = prediction_proba[le.transform(['N'])[0]]
    prob_approval = prediction_proba[le.transform(['Y'])[0]]

    # Display the result to the user
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

    st.subheader("Prediction Result:")
    st.success(result)
    st.write(f"**Probability of Approval:** {prob_approval:.2f}")
    st.write(f"**Probability of Rejection:** {prob_rejection:.2f}")
