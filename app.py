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
    # Create a DataFrame from the user inputs to perform feature engineering and encoding
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    
    # Perform feature engineering exactly as done in the training script
    input_df['LoanAmount_log'] = np.log(input_df['LoanAmount'])
    input_df['Total_Income'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']

    # Handle one-hot encoding for categorical features
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols)
    input_df_encoded = input_df_encoded.replace({True: 1, False: 0})
    
    # Reorder columns to match the scaler's fitted feature names
    # This is crucial for consistency.
    try:
        # Load the feature names from the scaler
        # Note: You can also save the feature names to a separate file during training for robustness.
        feature_names_in_order = scaler.feature_names_in_
        
        # Ensure all expected columns are present, fill with 0 if a category wasn't selected
        # for a given column (e.g., if you don't have a 'Dependents_3+' option, it won't be in the df)
        for col in feature_names_in_order:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0
                
        # Reorder the DataFrame columns
        X_input = input_df_encoded[feature_names_in_order]

    except AttributeError:
        # Fallback if the scaler was saved in an older scikit-learn version
        # You will need to manually list the columns in the correct order here.
        st.warning("Could not retrieve feature names from the scaler. Using a default column order. This might lead to incorrect predictions.")
        X_input = input_df_encoded
    except KeyError as e:
        st.error(f"Error: Missing a column required by the model. Check your one-hot encoding logic. Missing column: {e}")
        st.stop()

    # Scale the input data using the loaded scaler
    try:
        X_input_scaled = scaler.transform(X_input)
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
