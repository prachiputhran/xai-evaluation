# XAI Evaluation Project (SHAP + LIME)

## 📌 Overview
This project evaluates machine learning model interpretability using Explainable AI techniques such as **SHAP** and **LIME** across multiple real-world datasets.

The goal is to compare model performance and explanation stability across different ML models and datasets.

---

## 📊 Datasets Used
- Breast Cancer Wisconsin
- Heart Disease (UCI)
- German Credit Dataset

---

## 🤖 Models Implemented
- Logistic Regression
- Random Forest
- Neural Network (MLP)

---

## 🧠 Explainability Methods
- SHAP (Shapley Additive Explanations)
- LIME (Local Interpretable Model-agnostic Explanations)

---

## 📈 Evaluation Metrics
- Accuracy
- F1 Score
- SHAP Consistency
- SHAP Stability
- LIME Consistency
- LIME Stability
- Rank Correlation

---

## 📁 Output Files
All outputs are saved in the `xai_results/` folder:

- Model performance charts
- SHAP summary plots
- LIME analysis outputs
- Consistency & stability comparisons
- CSV metrics table

---

## 🚀 How to Run

```bash
python -m venv venv
venv\Scripts\activate
pip install shap lime scikit-learn pandas numpy matplotlib seaborn scipy ucimlrepo
python xai_evaluation.py