
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Preprocessing function
def preprocess_new_data(input_data, scaler, label_encoding_info):
    """
    Preprocess new data for prediction using saved preprocessing objects
    """
    # Create a DataFrame from the input data
    new_df = pd.DataFrame([input_data])
    
    # Handle missing values
    for col in new_df.columns:
        if new_df[col].dtype in [np.int64, np.float64]:
            new_df[col] = new_df[col].fillna(new_df[col].median())
        else:
            new_df[col] = new_df[col].fillna(new_df[col].mode()[0])
    
    # Apply label encoding to categorical columns using the saved encoding info
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_columns:
        if col in new_df.columns:
            # Create a mapping from the saved encoding info
            mapping = label_encoding_info[col]
            
            # Apply the mapping
            new_df[col] = new_df[col].map(mapping)
            # Fill any NaN values (for unseen categories) with the most frequent category
            if new_df[col].isna().any():
                first_value = list(mapping.values())[0]
                new_df[col] = new_df[col].fillna(first_value)
    
    # Scale the numerical features
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                          'Loan_Amount_Term', 'Credit_History']
    
    # Ensure all numerical features are present and convert to float
    for feature in numerical_features:
        if feature in new_df.columns:
            new_df[feature] = new_df[feature].astype(float)
    
    new_df[numerical_features] = scaler.transform(new_df[numerical_features])
    
    return new_df

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoding_info = joblib.load('label_encoding_info.pkl')
        return model, scaler, label_encoding_info
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.error("Please run the training code first to create model files.")
        return None, None, None

model, scaler, label_encoding_info = load_model()

st.title('Loan Approval Prediction System')
st.write('This app predicts whether a loan application will be approved based on applicant information.')

# Create input fields for user data
st.header('Applicant Information')

col1, col2 = st.columns(2)

with col1:
    loan_id = st.text_input('Loan ID', 'LP001015')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])

with col2:
    applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0.0, value=0.0)
    loan_amount = st.number_input('Loan Amount', min_value=0.0, value=100.0)
    loan_amount_term = st.number_input('Loan Amount Term', min_value=0.0, value=360.0)
    credit_history = st.selectbox('Credit History', [1.0, 0.0])
    property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Create a dictionary with the input data
input_data = {
    'Loan_ID': loan_id,
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}

# Preprocess the input data and make prediction
if st.button('Predict Loan Approval'):
    if model is None or scaler is None or label_encoding_info is None:
        st.error("Model not loaded. Please train the model first.")
    else:
        try:
            # Preprocess the input data
            processed_data = preprocess_new_data(input_data, scaler, label_encoding_info)
            
            # Remove Loan_ID as it's not used for prediction
            if 'Loan_ID' in processed_data.columns:
                processed_data = processed_data.drop('Loan_ID', axis=1)
            
            # Make prediction
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)
            
            # Display results
            st.subheader('Prediction Results')
            
            if prediction[0] == 1:
                st.success('Loan Approved! ✅')
                st.balloons()
            else:
                st.error('Loan Not Approved ❌')
            
            st.write(f'**Probability of Approval:** {prediction_proba[0][1] * 100:.2f}%')
            st.write(f'**Probability of Rejection:** {prediction_proba[0][0] * 100:.2f}%')
            
            # Show feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader('Top Influencing Factors')
                feature_importance = pd.DataFrame({
                    'feature': processed_data.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.bar_chart(feature_importance.set_index('feature').head(5))
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Add training section
st.header('Train Model')
if st.button('Train Model Now'):
    with st.spinner('Training model...'):
        try:
            # Your training code here
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            # Load and preprocess data
            def clean_df(df):
                df = df.replace({np.nan: None})
                return df
            
            # Load the data
            df_applicant = pd.read_json("applicant_info.json", lines=True)
            df_financial = pd.read_json("financial_info.json", lines=True)
            df_loan = pd.read_json("loan_info.json", lines=True)
            
            # Preprocess
            df_applicant = clean_df(df_applicant)
            df_financial = clean_df(df_financial)
            df_loan = clean_df(df_loan)
            
            df = df_applicant.merge(df_financial, on='Loan_ID', how='inner')
            df = df.merge(df_loan, on='Loan_ID', how='inner')
            
            # Handle missing values
            for col in df.columns:
                if df[col].dtype in [np.int64, np.float64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            
            # Encode categorical variables
            cat_cols = df.select_dtypes(include=['object']).columns
            le = {}
            for col in cat_cols:
                df[col] = df[col].astype(str)
                df[col] = df[col].fillna(df[col].mode()[0])
                le[col] = {val: idx for idx, val in enumerate(df[col].unique())}
                df[col] = df[col].map(le[col])
            
            # Prepare features and target
            X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
            y = df['Loan_Status']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale numerical features
            scaler = StandardScaler()
            numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                  'Loan_Amount_Term', 'Credit_History']
            X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
            X_test[numerical_features] = scaler.transform(X_test[numerical_features])
            
            # Train the model
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train, y_train)
            
            # Save the model and preprocessing objects
            joblib.dump(rf_classifier, 'model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(le, 'label_encoding_info.pkl')
            
            st.success("Model trained and saved successfully!")
            st.info("Please refresh the page to load the new model.")
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

# Add some information about the model
st.header('About the Model')
st.write('This model uses a Random Forest classifier trained on historical loan data.')
st.write('The model considers various factors including income, credit history, and property area to make predictions.')

# Add some tips for improving loan approval chances
st.header('Tips for Improving Loan Approval Chances')
st.write('''
- Maintain a good credit history
- Have a stable source of income
- Lower your debt-to-income ratio
- Provide accurate and complete documentation
- Consider applying with a co-applicant if needed
''')

