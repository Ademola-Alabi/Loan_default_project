import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st
from joblib import load

model = load("loan_defaul_model.joblib")

def main():
    st.title('Loan Prediction App')
    st.subheader('Provide Information')

    # Numeric inputs
    age = st.number_input('Age', min_value=18, max_value=100)
    income = st.number_input('Income')
    loan_amount = st.number_input('Loan Amount')
    credit_score = st.number_input('Credit Score')
    months_employed = st.number_input('Months Employed')
    num_credit_lines = st.number_input('Number of Credit Lines')
    interest_rate = st.number_input('Interest Rate')
    loan_term = st.number_input('Loan Term')
    dti_ratio = st.number_input('DTI Ratio')

    # Categorical inputs
    education = st.selectbox('Education', ['Select', 'High School', "Bachelor's", 'Master', 'PhD'])
    employment_type = st.selectbox('Employment Type', ['Select', 'Part-time', 'Full-time', 'Self-employed', 'Unemployed'])
    marital_status = st.selectbox('Marital Status', ['Select', 'Single', 'Married', 'Divorced'])
    has_mortgage = st.selectbox('Has Mortgage', ['Select', 'Yes', 'No'])
    has_dependents = st.selectbox('Has Dependents', ['Select', 'Yes', 'No'])
    loan_purpose = st.selectbox('Loan Purpose', ['Select', 'Personal', 'Education', 'Home', 'Car', 'Other'])
    has_cosigner = st.selectbox('Has CoSigner', ['Select', 'Yes', 'No'])

    if st.button('Predict'):
        # Prepare input data for prediction
        data = {
            'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
            'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
            'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
            # Handle the 'Select' option for categorical inputs; they should not contribute to dummy variables
            **{f'Education_{education}': 1 if education != 'Select' else 0},
            **{f'EmploymentType_{employment_type}': 1 if employment_type != 'Select' else 0},
            **{f'MaritalStatus_{marital_status}': 1 if marital_status != 'Select' else 0},
            **{f'HasMortgage_{has_mortgage}': 1 if has_mortgage == 'Yes' else 0},
            **{f'HasDependents_{has_dependents}': 1 if has_dependents == 'Yes' else 0},
            **{f'LoanPurpose_{loan_purpose}': 1 if loan_purpose != 'Select' else 0},
            **{f'HasCoSigner_{has_cosigner}': 1 if has_cosigner == 'Yes' else 0},
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])
        # Assuming a function to align and prepare df before prediction
        prepared_df = prepare_input(df)
        prediction = model.predict(prepared_df)
        st.write(f'Prediction: {prediction}')
def prepare_input(df):
    # This function should align the input df with the model's expected input
    # including handling missing dummy columns and ordering
    # For simplicity, this is left as a placeholder
    return df

if __name__ == '__main__':
    main()   