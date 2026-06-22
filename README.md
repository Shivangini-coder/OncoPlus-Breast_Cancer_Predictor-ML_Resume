# 🩺 OncoPlus – Breast Cancer Prediction using Machine Learning

## 📌 Overview

OncoPlus is a machine learning-based web application that predicts whether a breast tumor is **Benign** or **Malignant** using clinical diagnostic measurements from the Breast Cancer Wisconsin Dataset.

The project follows a complete machine learning workflow including feature selection, outlier detection, model comparison, evaluation, model serialization, and deployment. Multiple classification algorithms were evaluated, and **Logistic Regression** achieved the best overall performance and was selected for deployment.

---

## 🚀 Live Demo

**Deployed Application:**
https://huggingface.co/spaces/Shivangini2006/Breast_Cancer_Prediction_ML

---

## 🎯 Problem Statement

Breast cancer is one of the most common cancers affecting women worldwide. Early and accurate diagnosis significantly improves treatment outcomes.

This project aims to assist in diagnosis by predicting whether a tumor is malignant or benign using machine learning models trained on diagnostic features extracted from breast mass images.

---

## 📊 Dataset

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

The dataset contains measurements computed from digitized images of breast mass cell nuclei.

### Target Variable

| Label | Meaning   |
| ----- | --------- |
| 0     | Benign    |
| 1     | Malignant |

### Initial Features

30 numerical diagnostic features describing characteristics such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal Dimension

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing

* Removed irrelevant columns (`id`, `Unnamed: 32`)
* Encoded diagnosis labels into numerical format
* Separated feature matrix and target variable

### 2. Feature Selection using RFECV

Recursive Feature Elimination with Cross Validation (RFECV) was performed using Logistic Regression as the base estimator.

This reduced the original feature space and retained the most informative predictors.

**Optimal Features Selected:** 14

### 3. Outlier Detection using Local Outlier Factor (LOF)

Local Outlier Factor (LOF) was used to identify anomalous observations.

* LOF score threshold: **-2.5**
* Detected outliers were removed before model training

### 4. Feature Scaling

StandardScaler was applied to normalize feature values before training.

### 5. Model Training and Comparison

The following classification models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* XGBoost

---

## 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 98.25%   | 100.00%   | 95.24% | 97.56%   | 99.12%  |
| XGBoost             | 97.66%   | 100.00%   | 93.65% | 96.72%   | 98.68%  |
| Random Forest       | 97.08%   | 100.00%   | 92.06% | 95.87%   | 99.16%  |
| SVM                 | 97.08%   | 98.33%    | 93.65% | 95.94%   | 99.32%  |
| KNN                 | 95.32%   | 98.25%    | 88.89% | 93.33%   | 97.93%  |
| Decision Tree       | 95.32%   | 96.61%    | 90.48% | 93.44%   | 94.31%  |

### 🏆 Best Model

**Logistic Regression**

* Accuracy: **98.25%**
* Precision: **100%**
* Recall: **95.24%**
* F1 Score: **97.56%**
* ROC-AUC: **99.12%**

The Logistic Regression model achieved the best balance between accuracy, precision, recall, and interpretability, making it the final deployed model.

---

## 📊 Visualizations Included

The project includes:

* Class Distribution Analysis
* Correlation Analysis
* RFECV Feature Selection Curve
* Local Outlier Factor (LOF) Visualization
* Model Accuracy Comparison
* Confusion Matrix
* ROC-AUC Evaluation
* Logistic Regression Feature Influence Analysis

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib

### Deployment

* Streamlit
* Hugging Face Spaces

---

## 📂 Project Structure

```text
├── notebook.ipynb
├── best_model_LR.pkl
├── scaler.pkl
├── assets/
│   └── features.json
├── app.py
├── requirements.txt
└── README.md
```

---

## 💾 Model Serialization

The deployed application uses:

* `best_model_LR.pkl`
* `scaler.pkl`
* `assets/features.json`

These artifacts ensure that the deployment pipeline uses the exact preprocessing and feature set used during training.

---

## 🔮 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation based optimization
* SHAP-based model explainability
* Comparison with balancing techniques such as SMOTE
* Ensemble model experimentation

---

## 👩‍💻 Author

**Shivangini Gupta**

LinkedIn:
https://www.linkedin.com/in/shivangini-gupta-igdtuw/

---

### ⭐ If you found this project useful, consider starring the repository.
