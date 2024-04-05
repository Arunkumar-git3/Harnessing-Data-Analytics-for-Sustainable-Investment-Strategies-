import streamlit as st
import pandas as pd
import joblib

def load_model():
    model = joblib.load(r'C:\Users\asuss\Downloads\predict.pkl')
    return model


def predict_esg_risk(model, environment_risk, governance_risk, social_risk):
    sample = pd.DataFrame({
        'Environment Risk Score': [environment_risk],
        'Governance Risk Score': [governance_risk],
        'Social Risk Score': [social_risk]
    })
    prediction = model.predict(sample)[0]
    risk_levels = ["Low", "Negligible", "Medium", "High", "Severe"]
    prediction_text = risk_levels[prediction]
    return prediction_text

def main():
    st.title("ESG Risk Level Prediction")

    model = load_model()

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        company_names = data['Name'].tolist()
        selected_company = st.selectbox("Select a company", company_names)

        if selected_company:
            company_data = data[data['Name'] == selected_company].squeeze().to_dict()
            environment_risk = company_data['Environment Risk Score']
            governance_risk = company_data['Governance Risk Score']
            social_risk = company_data['Social Risk Score']

            prediction_text = predict_esg_risk(model, environment_risk, governance_risk, social_risk)
            st.markdown(f"The predicted ESG Risk Level for **{selected_company}** is **{prediction_text}**.")

    manual_input = st.checkbox("Enter data manually")

    if manual_input:
        with st.form("input_form"):
            st.subheader("Enter Company Details")
            environment_risk = st.number_input("Environment Risk Score", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
            governance_risk = st.number_input("Governance Risk Score", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
            social_risk = st.number_input("Social Risk Score", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
            submit_button = st.form_submit_button("Predict ESG Risk Level")

        if submit_button:
            prediction_text = predict_esg_risk(model, environment_risk, governance_risk, social_risk)
            st.markdown(f"The predicted ESG Risk Level is **{prediction_text}**.")

if __name__ == "__main__":
    main()