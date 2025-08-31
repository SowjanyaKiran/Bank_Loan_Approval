import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# --- 1. Load and prepare the data ---
# This is dummy data to make the script runnable. Replace with your actual data loading logic.
data = {
    'Loan_ID': range(1, 101),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Married': np.random.choice(['Yes', 'No'], 100),
    'Dependents': np.random.choice(['0', '1', '2', '3+'], 100),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], 100),
    'Self_Employed': np.random.choice(['Yes', 'No'], 100),
    'ApplicantIncome': np.random.randint(1500, 10000, 100),
    'CoapplicantIncome': np.random.randint(0, 5000, 100),
    'LoanAmount': np.random.randint(90, 700, 100),
    'Loan_Amount_Term': np.random.randint(240, 480, 100),
    'Credit_History': np.random.choice([1.0, 0.0], 100, p=[0.8, 0.2]),
    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], 100),
    'Loan_Status': np.random.choice(['Y', 'N'], 100, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# --- 2. Feature Engineering and Encoding ---
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Handle categorical features
df_encoded = pd.get_dummies(df.drop('Loan_ID', axis=1), columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
df_encoded = df_encoded.replace({True: 1, False: 0})

# --- 3. Split data ---
le = LabelEncoder()
df_encoded['Loan_Status_Encoded'] = le.fit_transform(df_encoded['Loan_Status'])

X = df_encoded.drop(['Loan_Status', 'Loan_Status_Encoded'], axis=1)
y = df_encoded['Loan_Status_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Scale numerical features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train and optimize the model with GridSearchCV ---
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

print("Model training complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# --- 6. Save the model, scaler, and label encoder ---
# Ensure this file name matches the one in app.py
with open('loan_approval_model.pkl', 'wb') as f:
    pickle.dump((best_model, scaler, le), f)

print("Model, scaler, and label encoder saved to 'loan_approval_model.pkl'")
