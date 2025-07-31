import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("/Users/suvedharam/payroll-prediction/model/rf_model.pkl")
# Load saved residuals
residuals_df = pd.read_csv('/Users/suvedharam/payroll-prediction/model/oof_residuals.csv')
abs_residuals_train = np.abs(residuals_df['residual'])

# List of valid job titles
job_titles = [
    "Displaced Disaster Worker", "EMT Paramedic", "Exception Job Code", "Fire Captain",
    "Firefighter", "Laborer", "Library Page (20 hrs)", "Library Technician I",
    "Library Technician I (10 hrs)", "Maintenance Worker I", "Maintenance Worker II",
    "Police Captain", "Police Lieutenant", "Police Officer", "Police Sergeant",
    "School Crossing Guard", "Senior Clerical Specialist"
]

st.set_page_config(page_title="Salary Anomaly Detector", layout="centered")
st.title("💼 Salary Anomaly Detection App")

st.markdown("Enter employee details to predict expected salary and identify anomalies.")

# --- User Input Form ---
with st.form("salary_form"):
    st.subheader("Employee Attributes")

    job_title = st.selectbox("Job Title", job_titles)
    pay_grade = st.number_input("Pay Grade", min_value=0, step=1)
    pay_step = st.number_input("Pay Step", min_value=0, step=1)
    scheduled_hours = st.number_input("Scheduled Hours", min_value=0.0)
    longevity_percentage = st.number_input("Longevity Percentage", min_value=0.0)
    
    st.subheader("Compensation")
    base_hourly_rate = st.number_input("Base Hourly Rate", min_value=0.0)
    total_hourly_rate = st.number_input("Total Hourly Rate", min_value=0.0)
    overtime_hourly_rate = st.number_input("Overtime Hourly Rate", min_value=0.0)

    annual_salary_input = st.number_input("Actual Annual Salary", min_value=0.0, step=100.0)

    # Let user select percentile threshold
    percentile = st.slider("Select anomaly detection percentile", 90, 99, 99)
    threshold = np.percentile(abs_residuals_train, percentile)

    submitted = st.form_submit_button("Predict and Check Anomaly")

# --- Inference Logic ---
if submitted:
    # Derived feature
    overtime_ratio = overtime_hourly_rate / base_hourly_rate if base_hourly_rate > 0 else 0.0

    # Base input features
    input_dict = {
        "pay_grade": pay_grade,
        "pay_step": pay_step,
        "scheduled_hours": scheduled_hours,
        "longevity_percentage": longevity_percentage,
        "total_hourly_rate": total_hourly_rate,
        "overtime_ratio": overtime_ratio 
    }

    # Add one-hot encoded job title columns (all set to 0 first)
    for title in job_titles:
        col_name = f"job_title_{title}"
        input_dict[col_name] = 1 if job_title == title else 0

    input_df = pd.DataFrame([input_dict])

    # Predict log salary and convert back
    predicted_log_salary = model.predict(input_df)[0]
    predicted_salary = np.expm1(predicted_log_salary)

    # Residual and anomaly check
    actual_log_salary = np.log1p(annual_salary_input)
    residual = actual_log_salary - predicted_log_salary
    residual_dollar = annual_salary_input - predicted_salary

    abs_residual = abs(residual)

    # Compute bounds in log space and convert to dollars
    lower_log = predicted_log_salary - threshold
    upper_log = predicted_log_salary + threshold
    lower_dollar = np.exp(lower_log)
    upper_dollar = np.exp(upper_log)




    # Show deviation ---
    deviation_ratio = annual_salary_input / predicted_salary
    percent_diff = (deviation_ratio - 1) * 100
    st.write(f"📉 Deviation from predicted: {percent_diff:+.2f}%")

    # --- Output ---
    st.markdown("---")
    st.subheader("📊 Results")
    st.write(f"**Predicted Salary:** ${predicted_salary:,.2f}")
    st.write(f"**Actual Salary:** ${annual_salary_input:,.2f}")
    st.write(f"Residual (dollar difference): ${residual_dollar:,.2f}")
    st.write(f"💡 Expected salary range (based on {percentile}th percentile):")
    st.write(f"📌 ${lower_dollar:,.2f}  -  ${upper_dollar:,.2f}")
    

    # Anomaly detection
    if abs_residual > threshold:
        st.error("🚨 Anomaly Detected! Salary is outside expected range.")
    else:
        st.success("✅ Salary is within expected range.")
