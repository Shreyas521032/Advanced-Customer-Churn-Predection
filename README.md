# 📉 Advanced Customer Churn Prediction

## 📋 Overview
This project implements a machine learning solution to predict customer churn using three different algorithms: **Random Forest**, **XGBoost**, and **Logistic Regression**. The model helps identify customers who are likely to discontinue services, enabling proactive retention strategies.

🌍 **Live Project**: [https://shreyas-advanced-customer-churn-predection.streamlit.app](https://shreyas-advanced-customer-churn-predection.streamlit.app/)

## 📑 Table of Contents
- [✨ Features](#-features)
- [📦 Requirements](#-requirements)
- [📁 Project Structure](#-project-structure)
- [🛠️ Installation](#-installation)
- [📊 Model Comparison](#-model-comparison)

## ✨ Features
- 🔄 Data preprocessing and feature engineering  
- 🤖 Implementation of three machine learning algorithms:
  - 🌲 Random Forest Classifier
  - ⚡ XGBoost Classifier
  - ➕ Logistic Regression
- 📈 Model performance comparison and evaluation
- 🧠 Feature importance analysis
- 🔁 Cross-validation for robust model validation

## Requirements

```
python>=3.8
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

## Project Structure
```
customer-churn-prediction/
│
├── data/
│   ├── Churn_Modelling.csv
│  
├── saved models/
│   ├── Gradient_Boosting_Classifier.joblib
│   ├── scaler.joblib
│  
├── notebooks/
│   └── Model.ipynb
│
└── README.md
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Model Comparison

### Performance Metrics

| Model               | Accuracy | Precision | 
|--------------------|----------|-----------|
| Random Forest      | 0.87     | 0.83      | 
| XGBoost            | 0.88     | 0.86      | 
| Logistic Regression| 0.82     | 0.77      | 

### Key Findings
- XGBoost performed best overall with highest accuracy and AUC-ROC scores
- Random Forest showed comparable performance with slightly lower metrics
- Logistic Regression provided a good baseline but was outperformed by both ensemble methods

Threshold vs Recall and Threshold vs Precision graph (XGBoost)


![image](https://github.com/user-attachments/assets/42be4ba5-052d-4e7c-8c16-be57bc929d80)

ROC Curve

![{05BAA722-2B5D-466B-94C4-5ECB09D9A904}](https://github.com/user-attachments/assets/3a3cacb5-15e2-4876-bf49-94d3d3515866)

## License

© 2025 Shreyas Kasture

All rights reserved.

This software and its source code are the intellectual property of the author. Unauthorized copying, distribution, modification, or usage in any form is strictly prohibited without explicit written permission.

For licensing inquiries, please contact: shreyas200410@gmail.com

