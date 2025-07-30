import streamlit as st
import numpy as np
import joblib
import pandas as pd
import math

# === Load model ===
try:
    model = joblib.load("/Users/suvedharam/payroll-prediction/model/rf_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please check the path.")
    st.stop()

# === Set up the residual std threshold ===
residual_std = 0.076563  # Your actual value from training (in log space)
threshold_multiplier = st.sidebar.selectbox("Anomaly Threshold (×std)", [2, 3], index=1)
threshold = threshold_multiplier * residual_std

st.title("🏢 Salary Prediction & Anomaly Detection App")
st.markdown("---")

# === Create two columns for better layout ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Employee Information")
    
    # === Numerical input sliders ===
    total_hourly_rate = st.slider("Total Hourly Rate ($)", 10.0, 120.0, 30.0, step=1.0)
    scheduled_hours = st.slider("Scheduled Hours (per week)", 0.0, 125.0, 40.0, step=1.0)
    overtime_ratio = st.slider("Overtime Ratio", 0.0, 2.0, 0.1, step=0.01)
    pay_grade = st.slider("Pay Grade", 100.0, 8081.0, 2500.0, step=100.0)
    pay_step = st.slider("Pay Step", 1.0, 21.0, 5.0, step=1.0)
    longevity_percentage = st.slider("Longevity Percentage", 0.0, 0.20, 0.0, step=0.01)

with col2:
    st.subheader("👔 Job Details")
    
    # === Job Title Dropdown ===
    job_titles = [
        "Displaced Disaster Worker", "EMT Paramedic", "Exception Job Code", "Fire Captain",
        "Firefighter", "Laborer", "Library Page (20 hrs)", "Library Technician I",
        "Library Technician I (10 hrs)", "Maintenance Worker I", "Maintenance Worker II",
        "Police Captain", "Police Lieutenant", "Police Officer", "Police Sergeant",
        "School Crossing Guard", "Senior Clerical Specialist"
    ]
    
    selected_job = st.selectbox("Job Title", job_titles, index=job_titles.index("Police Officer"))
    
    # === Actual salary input ===
    actual_salary = st.number_input("Actual Annual Salary ($)", min_value=0, value=50000, step=1000)

# === One-hot encode job title ===
job_title_encoded = [1 if title == selected_job else 0 for title in job_titles]

# === Construct input vector for model ===
X_input = np.array([
    pay_grade, pay_step, scheduled_hours,
    longevity_percentage, total_hourly_rate,
    overtime_ratio
] + job_title_encoded).reshape(1, -1)

st.markdown("---")

# === Make prediction when button is clicked ===
if st.button("🔮 Predict Salary & Check for Anomalies", type="primary"):
    
    # === Predict salary (model returns log-transformed value) ===
    predicted_log_salary = model.predict(X_input)[0]
    predicted_salary = math.exp(predicted_log_salary)  # Convert back from log space
    
    # === Calculate residual in log space (correct approach) ===
    actual_log_salary = math.log(actual_salary)
    residual_log = abs(actual_log_salary - predicted_log_salary)
    
    # === Display results ===
    col1_result, col2_result = st.columns(2)
    
    with col1_result:
        st.subheader("📈 Prediction Results")
        st.metric("Predicted Salary", f"${predicted_salary:,.0f}")
        st.metric("Actual Salary", f"${actual_salary:,.0f}")
        
        # Calculate percentage difference
        pct_diff = ((actual_salary - predicted_salary) / predicted_salary) * 100
        st.metric("Difference", f"{pct_diff:+.1f}%")
    
    with col2_result:
        st.subheader("🚨 Anomaly Detection")
        
        # === Anomaly Detection (in log space) ===
        if residual_log > threshold:
            st.error(f"🚨 **ANOMALY DETECTED!**")
            st.error(f"Log residual: {residual_log:.4f} > threshold: {threshold:.4f}")
            
            # Provide context in dollar terms
            dollar_diff = abs(actual_salary - predicted_salary)
            st.error(f"Salary differs by ${dollar_diff:,.0f} ({pct_diff:+.1f}%)")
            
            # Suggest possible reasons
            st.markdown("**Possible reasons:**")
            st.markdown("- Data entry error")
            st.markdown("- Special compensation package")
            st.markdown("- Incorrect job classification")
            st.markdown("- Performance bonus/penalty")
            
        else:
            st.success("✅ **Salary within expected range**")
            st.success(f"Log residual: {residual_log:.4f} ≤ threshold: {threshold:.4f}")
            st.info(f"Salary differs by ${abs(actual_salary - predicted_salary):,.0f} ({abs(pct_diff):.1f}%)")
    
    # === Additional insights ===
    st.markdown("---")
    st.subheader("📊 Additional Insights")
    
    # Create metrics for key insights
    col1_insight, col2_insight, col3_insight = st.columns(3)
    
    with col1_insight:
        annual_from_hourly = total_hourly_rate * scheduled_hours * 52
        st.metric("Expected from Hourly Rate", f"${annual_from_hourly:,.0f}")
    
    with col2_insight:
        if overtime_ratio > 1.0:
            overtime_premium = (overtime_ratio - 1.0) * 100
            st.metric("Overtime Premium", f"{overtime_premium:.1f}%")
        else:
            st.metric("Overtime Premium", "0%")
    
    with col3_insight:
        st.metric("Residual Std Used", f"{residual_std:.4f}")

# === Sidebar information ===
st.sidebar.markdown("### ℹ️ Model Information")
st.sidebar.info(f"""
**Model Type:** Random Forest
**Residual Std:** {residual_std:.4f} (log space)
**Threshold:** {threshold_multiplier}× std = {threshold:.4f}
**Features:** {6 + len(job_titles)} total
""")

st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown("""
1. **Adjust employee parameters** using the sliders
2. **Select job title** from dropdown
3. **Enter actual salary** for comparison
4. **Click 'Predict'** to see results
5. **Review anomaly status** and insights
""")

