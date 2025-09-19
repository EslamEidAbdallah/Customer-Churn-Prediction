import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load saved objects
# -------------------------
model = joblib.load("Model/telco_churn_logreg_model.pkl")
scaler = joblib.load("Model/telco_churn_scaler.pkl")
encoders = joblib.load("Model/telco_churn_encoders.pkl")

# Extract encoders
gender_le = encoders.get("Gender")
subtype_le = encoders.get("Subscription Type")
contract_le = encoders.get("Contract Length")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊", layout="wide")
st.title("📞 Telco Customer Churn Prediction App")

tab1, tab2 = st.tabs(["📥 Data Input", "📊 Prediction Result"])

numerical_features = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                      "Payment Delay", "Total Spend", "Last Interaction"]

with tab1:
    st.subheader("Enter customer information")
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, format="%.2f")
        Gender = st.selectbox("Gender", gender_le.classes_)
        Tenure = st.number_input("Tenure (months)", min_value=0.0, max_value=120.0, value=12.0, format="%.2f")
        Usage_Frequency = st.number_input("Usage Frequency", min_value=0.0, max_value=1000.0, value=10.0, format="%.2f")
        Support_Calls = st.number_input("Support Calls", min_value=0.0, max_value=1000.0, value=1.0, format="%.2f")

    with col2:
        Payment_Delay = st.number_input("Payment Delay (days)", min_value=0.0, max_value=365.0, value=0.0, format="%.2f")
        Subscription_Type = st.selectbox("Subscription Type", subtype_le.classes_)
        Contract_Length = st.selectbox("Contract Length", contract_le.classes_)
        Total_Spend = st.number_input("Total Spend", min_value=0.0, max_value=1_000_000.0, value=500.0, format="%.2f")
        Last_Interaction = st.number_input("Last Interaction (days since)", min_value=0.0, max_value=10000.0, value=7.0, format="%.2f")

    if st.button("✅ Predict"):
        input_df = pd.DataFrame([{
            "Age": Age,
            "Gender": Gender,
            "Tenure": Tenure,
            "Usage Frequency": Usage_Frequency,
            "Support Calls": Support_Calls,
            "Payment Delay": Payment_Delay,
            "Subscription Type": Subscription_Type,
            "Contract Length": Contract_Length,
            "Total Spend": Total_Spend,
            "Last Interaction": Last_Interaction
        }])

        # Encode categoricals
        input_df["Gender"] = gender_le.transform(input_df["Gender"])
        input_df["Subscription Type"] = subtype_le.transform(input_df["Subscription Type"])
        input_df["Contract Length"] = contract_le.transform(input_df["Contract Length"])

        # Scale numerical features
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Predict directly
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.session_state["prediction"] = int(pred)
        st.session_state["probability"] = float(prob)
        st.success("Prediction complete! Go to Result tab.")

with tab2:
    st.subheader("Prediction Result")
    if "prediction" in st.session_state:
        p = st.session_state["prediction"]
        pr = st.session_state["probability"]
        if p == 1:
            st.error(f"⚠ Customer likely to churn — probability: {pr*100:.2f}%")
        else:
            st.success(f"✅ Customer likely to stay — probability of staying: {(1-pr)*100:.2f}%")
    else:
        st.info("No prediction yet. Please enter data and click Predict.")

