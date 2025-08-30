
import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Credit Scoring App")
st.title("💳 Creditworthiness Prediction App")

try:
    model = joblib.load("models/RandomForest.pkl")
except:
    st.error("Model not found. Train it and place it in 'models/RandomForest.pkl'")
    st.stop()

income = st.number_input("Annual Income", min_value=0, step=1000)
debt = st.number_input("Outstanding Debt", min_value=0, step=500)
employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"])
credit_history = st.slider("Credit History (years)", 0, 30)
debt_to_income = debt / (income + 1)
emp_status_encoded = 1 if employment_status == "Employed" else 0

input_df = pd.DataFrame([[income, debt, emp_status_encoded, credit_history, debt_to_income]],
                        columns=["income", "debt", "employment_status", "credit_history", "debt_to_income"])

if st.button("Predict Creditworthiness"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    label = "Creditworthy ✅" if prediction == 1 else "Not Creditworthy ❌"
    st.success(f"Prediction: {label}")
    st.info(f"Confidence Score: {proba:.2%}")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Credit Scoring Report", ln=True)
    for col in input_df.columns:
        pdf.cell(200, 10, txt=f"{col}: {input_df[col][0]}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {proba:.2%}", ln=True)

    if not os.path.exists("reports"):
        os.makedirs("reports")
    pdf.output("reports/prediction_report.pdf")
    st.info("📄 PDF saved in 'reports/'")

    try:
        st.subheader("🔍 SHAP Explanation")
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=5, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation error: {e}")

    try:
        st.subheader("📊 Feature Importance")
        importance = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': input_df.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        st.bar_chart(feat_df.set_index("Feature"))
    except:
        st.warning("Feature importance not available.")
