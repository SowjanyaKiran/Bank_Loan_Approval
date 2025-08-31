# app.py
import streamlit as st
import pickle
import numpy as np
import base64

# ============ Custom CSS for a beautiful look ============
def add_bg_and_style():
    # Use a high-quality background image URL or a local file encoded to base64
    # Example using a URL
    # bg_image_url = "https://images.unsplash.com/photo-1518655049386-3e9114b3084f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80"
    
    # Or for a local image, encode it to base64
    # with open("path/to/your/image.jpg", "rb") as image_file:
    #     encoded_image = base64.b64encode(image_file.read()).decode()
    # bg_image_base64 = f"data:image/jpeg;base64,{encoded_image}"

    # For this example, let's use a simple background gradient
    custom_css = """
    <style>
    .stApp {
        background-color: #f0f2f6;
        background-image: linear-gradient(to right top, #a6c0fe, #c4a9f9, #e094f3, #f77ee6, #ff69d8);
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .css-1jc7-133a {
        color: white;
    }
    
    .main .block-container {
        padding: 4rem 1rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    h1 {
        text-align: center;
        color: #4A90E2;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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
    st.markdown(custom_css, unsafe_allow_html=True)

# ========== 1. Load Pickle Model ==========
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, le = pickle.load(f)

# Add styling to the app
add_bg_and_style()

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
    input_data = [
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

    # Preprocess the input data
    input_data_as_array = np.array(input_data).reshape(1, -1)
    
    # Scale the numerical features (6 to 9)
    # The scaler was trained on all features in the original notebook, so we apply it to all
    scaled_data = scaler.transform(input_data_as_array)
    
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.balloons()
        st.success("✅ Congratulations! Your loan is likely to be approved.")
    else:
        st.error("😔 We regret to inform you that your loan is likely to be rejected.")

    st.markdown("---")
    st.markdown(f"**Predicted Label:** `{le.inverse_transform(prediction)[0]}`")
