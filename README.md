# Predicting Payroll & Spotting Anomalies with Machine Learning

> Ensuring fairness, accuracy, and data-driven payroll decisions

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E.svg)](https://scikit-learn.org/)


##  Project Overview

This project addresses a critical challenge in payroll management for small and medium businesses (SMBs): ensuring fair, accurate, and consistent compensation across employees. By leveraging machine learning, we can predict expected salaries and detect anomalies that may indicate errors, bias, or compliance issues.

### The Problem

- **Inconsistent job titles** and outdated pay scales across SMBs
- **Manual payroll reviews** that are time-consuming and don't scale
- **Human error** leading to compensation discrepancies
- **Pattern detection** across multiple client companies is challenging

### The Solution

A machine learning pipeline that:
- Predicts employee salaries using compensation and job attributes
- Detects anomalies where actual pay deviates significantly from expected
- Provides explainable insights for fair compensation decisions
- Enables scalable, data-driven payroll reviews

##  Key Features

- **Salary Prediction**: Robust Random Forest model with 99.4% R² accuracy
- **Anomaly Detection**: Automated flagging of compensation outliers
- **Interactive Dashboard**: Streamlit web app for easy exploration
- **Model Explainability**: SHAP values for transparent decision-making
- **Scalable Architecture**: Designed for multi-client payroll processing

##  Dataset

**Source**: Baton Rouge city employee data (public domain via GitHub - Payroll Sample Tables for Teradata)

**Size**: ~10,000 employee records

### Features Used

| Category | Features |
|----------|----------|
| **Compensation** | Total hourly rate, Overtime ratio |
| **Job Attributes** | Job title, Pay grade, Pay step, Scheduled hours |
| **Experience** | Longevity percentage |

**Target Variable**: Annual Salary

## 🛠️ Technical Implementation

### Data Preprocessing
- Removed collinear features (base/overtime rates)
- Engineered overtime ratio feature
- Applied KNN imputation for missing pay grades/steps
- Filtered job titles to those with ≥200 records
- Removed statistical outliers

### Model Selection

| Model | R² Score | MSE (log scale) | Notes |
|-------|----------|-----------------|-------|
| Linear Regression | **0.969** | **0.013** | Baseline model |
| **Random Forest** | **0.994** | **0.051** | **Selected model** |
| XGBoost | **0.995** | **0.002** | High accuracy, more complex |

**Why Random Forest?**
- Captures non-linear relationships
- Handles feature interactions naturally
- Robust to outliers
- Less prone to overfitting
- Provides feature importance insights

### Anomaly Detection Method
1. Calculate residuals from out-of-fold predictions
2. Determine percentile-based thresholds
3. Flag observations exceeding thresholds
4. Generate reports for manual review

## 🎮 Interactive Demo

The project includes a Streamlit web application that allows users to:
- Input employee details and get salary predictions
- View model explanations via SHAP values
- Detect and explore salary anomalies
- Visualize compensation trends and patterns

## 📈 Results & Impact

### Model Performance
- **Cross-validated R² Score**: 0.985
- **Final tuned model R²**: 0.994
- **Robust predictions** across different employee segments

### Business Benefits
-  Ensure fairness & equity in compensation
-  Detect outliers, errors, and potential bias
-  Enable data-driven payroll reviews
-  Support audit readiness & compliance
-  Scale payroll analysis across multiple clients

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/suvedharam97/payroll-prediction.git
cd payroll-prediction
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run app.py
```

## 📁 Project Structure
```
payroll-prediction/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned data
├── notebooks/
│   ├── payroll_prediction.ipynb          
├── models/
│   └── oof_residuals.csv
    └── rf_model.pkl
    └── xgb_model.pkl # Saved model artifacts
|── models/
    ├── app.py                  # Streamlit application
├── requirements.txt
└── README.md
```

## 🔮 Future Enhancements

- **Feedback Loops**: Continuous model retraining with new payroll data
- **Extended Coverage**: Anomaly detection for bonuses and benefits
- **HR Integration**: Translate anomalies into actionable policy insights
- **Real-time Monitoring**: Live payroll anomaly alerts
- **Multi-client Dashboards**: Comparative analysis across organizations


## 👨‍💻 Author

**Suvedha Ram**

- 💼 LinkedIn: [linkedin.com/in/suvedha-ram](https://linkedin.com/in/suvedha-ram)
- 🐙 GitHub: [github.com/suvedharam97](https://github.com/suvedharam97)

## 🙏 Acknowledgments

- Baton Rouge city for providing the public domain employee dataset
- Teradata for the sample payroll table structure
- The open-source community for the amazing tools and libraries

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Built with ❤️ for fair and transparent payroll management*
