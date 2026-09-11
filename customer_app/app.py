import streamlit as st
import pandas as pd
import joblib
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)
# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------
model = joblib.load("logistic regression.pkl")
scaler = joblib.load("scaler.pkl")
# column.pkl contains the 30 columns expected by the model
with open("column.pkl", "rb") as file:
    columns = joblib.load(file)
# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("📊 Customer Churn Prediction")
st.write(
    "Enter the customer's information below to predict "
    "whether the customer is likely to churn."
)
st.divider()
# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------
st.header("👤 Customer Information")
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )
with col3:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )
col1, col2, col3 = st.columns(3)
with col1:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )
with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )
with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
col1, col2, col3 = st.columns(3)
with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
with col2:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
with col3:
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )
# --------------------------------------------------
# BILLING INFORMATION
# --------------------------------------------------
st.header("💳 Billing Information")
col1, col2, col3 = st.columns(3)
with col1:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )
with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
with col3:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )
col1, col2 = st.columns(2)
with col1:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )
with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )
st.divider()
# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):
    # ----------------------------------------------
    # CREATE ALL MODEL FEATURES
    # ----------------------------------------------
    input_data = pd.DataFrame(0, index=[0], columns=columns)
    # Numerical columns
    input_data["SeniorCitizen"] = (
        1 if senior_citizen == "Yes" else 0
    )
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    # ----------------------------------------------
    # GENDER
    # ----------------------------------------------
    if gender == "Male":
        input_data["gender_Male"] = 1
    # ----------------------------------------------
    # CUSTOMER DETAILS
    # ----------------------------------------------
    if partner == "Yes":
        input_data["Partner_Yes"] = 1
    if dependents == "Yes":
        input_data["Dependents_Yes"] = 1
    if phone_service == "Yes":
        input_data["PhoneService_Yes"] = 1
    # ----------------------------------------------
    # INTERNET SERVICE
    # ----------------------------------------------
    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_data["InternetService_No"] = 1
    # ----------------------------------------------
    # ONLINE SECURITY
    # ----------------------------------------------
    if online_security == "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif online_security == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1
    # ----------------------------------------------
    # TECH SUPPORT
    # ----------------------------------------------
    if tech_support == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif tech_support == "No internet service":
        input_data["TechSupport_No internet service"] = 1
    # ----------------------------------------------
    # CONTRACT
    # ----------------------------------------------
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1
    # ----------------------------------------------
    # PAPERLESS BILLING
    # ----------------------------------------------
    if paperless_billing == "Yes":
        input_data["PaperlessBilling_Yes"] = 1
    # ----------------------------------------------
    # PAYMENT METHOD
    # ----------------------------------------------
    if payment_method == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment_method == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1
    # ----------------------------------------------
    # SCALE DATA
    # ----------------------------------------------
    scaled_data = scaler.transform(input_data)
    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------
    prediction = model.predict(scaled_data)[0]
    # ----------------------------------------------
    # PREDICTION PROBABILITY
    # ----------------------------------------------
    probability = model.predict_proba(scaled_data)[0][1]
    st.divider()
    # ----------------------------------------------
    # RESULT
    # ----------------------------------------------
    if prediction == 1:
        st.error(
            "⚠️ Customer is likely to CHURN"
        )
        st.warning(
            f"Churn Probability: {probability:.2%}"
        )
        st.write(
            "The model predicts that this customer "
            "may leave the company."
        )
    else:
        st.success(
            "✅ Customer is likely to STAY"
        )
        st.info(
            f"Churn Probability: {probability:.2%}"
        )
        st.write(
            "The model predicts that this customer "
            "is unlikely to leave the company."
        )