# 📊 End-to-End Customer Churn Prediction using XGBoost & SMOTE

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)
![SMOTE](https://img.shields.io/badge/Technique-SMOTE-orange.svg)
![University](https://img.shields.io/badge/TU_Dublin-MSc_Data_Science-purple.svg)

## 📌 Project Overview
This project addresses the critical business challenge of customer retention. Using a dataset of telecom customers, I built a machine learning pipeline that predicts whether a customer will churn (cancel their service). 

The primary technical challenge in this project was **class imbalance**—a common real-world scenario where the number of retained customers far outweighs those who churn. I solved this using **SMOTE** within an **Imbalanced-Learn Pipeline** to ensure the model learned the characteristics of churners effectively without data leakage.

## 🎯 Key Objectives
* **Predictive Modeling:** Build a classifier to identify high-risk customers.
* **Class Imbalance Management:** Implement SMOTE to balance the training set.
* **Feature Engineering:** Automated preprocessing of numerical and categorical data using `ColumnTransformer`.
* **Business Insights:** Identify the top drivers of churn to inform marketing strategy.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **ML Libraries:** `Scikit-Learn`, `XGBoost`, `Imbalanced-Learn`
* **Visualization:** `Seaborn`, `Matplotlib`
* **Data Handling:** `Pandas`, `NumPy`

## 🚀 The Pipeline
To ensure production-grade code, I implemented a `Pipeline` architecture:
1. **Preprocessing:** * Numerical features (Tenure, MonthlyCharges, TotalCharges) are scaled via `StandardScaler`.
    * Categorical features are encoded via `OneHotEncoder`.
2. **Resampling:** SMOTE is applied to balance the `Churn` target variable.
3. **Classification:** An `XGBoost` classifier is trained on the balanced data.

## 📈 Results & Evaluation
The model is evaluated using metrics that matter for imbalanced datasets:
* **ROC-AUC Score:** Measures the model's ability to distinguish between classes.
* **Confusion Matrix:** Analyzes False Negatives (the "Costly Misses" where a churner is predicted as staying).
* **Feature Importance:** Highlights exactly which behaviors (like Contract type or Monthly Charges) lead to churn.

## 📂 Repository Structure
```bash
├── customer-churn.csv    # The raw dataset
├── project_1.py          # Main Python script with full ML pipeline
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
