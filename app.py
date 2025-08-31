# app.py
import streamlit as st
import pickle
import numpy as np
import base64

# ============ Custom CSS for a beautiful look ============
def add_bg_and_style():
    # Hero section with background image
    hero_section_css = """
    <style>
    .hero-container {
        background-image: url('https://images.unsplash.com/photo-1544377193-33e142e09641?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        border-radius: 15px;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 15px;
    }
    
    .hero-content {
        position: relative;
        z-index: 10;
        padding: 0 20px;
    }

    .hero-content h1 {
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .hero-content p {
        font-size: 1.2rem;
    }

    /* General app styling */
    .stApp {
        background-color: #f0f2f6;
        background-image: linear-gradient(to right top, #a6c0fe, #c4a9f9, #e094f3, #ff69d8, #ff69d8);
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .main .block-container {
        padding: 4rem 1rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton>button:hover {
        background-color: #5A9CEF;
        transform: scale(1.05);
    }
    
    .stSelectbox label, .stNumberInput label {
        color: #333;
        font-weight: 500;
    }

    .stTextInput>div>div>input, .stSelectbox>div>div, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 8px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    </style>
    """
    st.markdown(hero_section_css, unsafe_allow_html=True)

# ========== 1. Load Pickle Model ==========
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, le = pickle.load(f)

# Add styling to the app
add_bg_and_style()

# ========== Add Hero Section ==========
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <h1>Financial Freedom Starts Here</h1>
        <p>Your trusted partner for quick and reliable loan approval predictions.</p>
    </div>
</div>
""", unsafe_allow_html=True)


st.title("🏦 Loan Approval Prediction App")
st.markdown("Fill the applicant details and check your loan approval status.")

# ========== 2. User Inputs ==========
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.number_input("Co-applicant Income", min_value=0, step=100)
loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0, step=12)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# ========== 3. Manual Encoding ==========
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_emp_map = {"Yes": 1, "No": 0}
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

if st.button("Predict Loan Approval"):
    # Create a single list with all features in the correct order
    # The order of features here must match the order used to train the model.
    full_input_data = [
        gender_map[gender],
        married_map[married],
        dependents_map[dependents],
        education_map[education],
        self_emp_map[self_employed],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_map[property_area],
    ]
    
    # Convert to a NumPy array for processing
    full_input_data_array = np.array(full_input_data).reshape(1, -1)
    
    # Identify the numerical features and scale them
    numerical_features = full_input_data_array[:, 5:10]
    scaled_numerical_features = scaler.transform(numerical_features)
    
    # Combine the scaled numerical data with the unscaled categorical data
    final_input_array = np.hstack((
        full_input_data_array[:, :5],  # First 5 are categorical
        scaled_numerical_features,     # Scaled numerical features
        full_input_data_array[:, 10:]  # Last one is categorical (Property Area)
    ))

    prediction = model.predict(final_input_array)

    if prediction[0] == 1:
        st.balloons()
        st.success("✅ Congratulations! Your loan is likely to be approved.")
    else:
        st.error("😔 We regret to inform you that your loan is likely to be rejected.")

    st.markdown("---")
    st.markdown(f"**Predicted Label:** `{le.inverse_transform(prediction)[0]}`")
