# 📊 Telecom Customer Churn Prediction & Decision Intelligence

An end-to-end machine learning and explainability pipeline implementing the methodology from the research paper:  
**"Explaining customer churn prediction in telecom industry using tabular machine learning models"** (*Machine Learning with Applications*, Elsevier, 2024).

---

## 📌 Executive Summary

Customer acquisition costs 5x to 7x more than customer retention in the telecommunications industry. While standard classifiers can predict *who* might churn, retention teams require transparent explanations to understand *why* customers leave and design targeted business interventions.

This project replicates the benchmark workflow across 7 tabular machine learning classifiers, validates model performance using 10-fold stratified cross-validation, and surfaces root-cause attrition drivers via feature attribution.

---

## 🚀 Key Results & Benchmark

Across a stratified 10-fold cross-validation setup on ~7,000 subscriber records:

| Model | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GBM (Gradient Boosting)** | **0.85 ± 0.02** | **0.81 ± 0.02** | 0.67 ± 0.04 | 0.55 ± 0.03 | 0.60 ± 0.02 |
| **Logistic Regression** | 0.85 ± 0.03 | 0.79 ± 0.02 | 0.64 ± 0.04 | 0.47 ± 0.06 | 0.54 ± 0.05 |
| **AdaBoost** | 0.85 ± 0.01 | 0.79 ± 0.01 | 0.65 ± 0.02 | 0.50 ± 0.06 | 0.57 ± 0.03 |
| **Random Forest** | 0.84 ± 0.01 | 0.80 ± 0.02 | 0.71 ± 0.04 | 0.43 ± 0.08 | 0.53 ± 0.07 |
| **XGBoost** | 0.83 ± 0.02 | 0.80 ± 0.03 | 0.68 ± 0.01 | 0.55 ± 0.02 | 0.61 ± 0.03 |
| **SVC** | 0.80 ± 0.02 | 0.78 ± 0.01 | 0.68 ± 0.03 | 0.34 ± 0.02 | 0.45 ± 0.02 |
| **Neural Network (MLP)** | 0.77 ± 0.06 | 0.74 ± 0.06 | 0.58 ± 0.26 | 0.43 ± 0.31 | 0.41 ± 0.21 |

> **Performance Verdict:** Gradient Boosting Machine (GBM) achieved the highest overall discriminatory performance and accuracy, confirming the paper's findings that ensemble boosting algorithms outperform deep neural architectures on small-to-medium tabular customer data.

---

## 📈 Visual Dashboard

![Executive Dashboard](outputofcustomerchurn/visuals/executive_linkedin_dashboard.png)

### Key Analytical Takeaways:
1. **Tenure is the Primary Anchor:** Account longevity (`tenure`) is the single strongest negative predictor of churn. Flight risk is heavily concentrated in the first 6 months of customer onboarding.
2. **Contract Type Risk:** Month-to-month subscribers experience an attrition rate of **~43%**, whereas 1-year and 2-year contract holders drop to **~11%** and **<3%** respectively.
3. **Fiber Optic Friction:** Fiber optic internet accounts exhibit double the churn rate of DSL subscribers, signaling pricing sensitivity or onboarding technical friction.
4. **Payment Method Sensitivity:** Subscribers paying via manual electronic check are substantially more prone to churn than those on automated credit card or bank transfer billing.

---

## 📁 Repository Structure

```text
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset from Kaggle
├── customer_churn_analysis.ipynb          # End-to-end pipeline notebook
├── outputofcustomerchurn/
│   ├── metrics/
│   │   ├── 10_fold_cv_benchmark.csv      # Cross-validation summary
│   │   ├── holdout_test_metrics.csv      # 20% holdout test evaluation
│   │   └── classification_reports.txt    # Precision/Recall/F1 per model
│   └── visuals/
│       ├── executive_linkedin_dashboard.png
│       ├── gbm_confusion_matrix.png
│       ├── gbm_feature_importance.png
│       └── models_roc_curves.png
└── README.md
