# 🧠 ANN-Based Customer Churn Predictor

This project is a machine learning web application that predicts whether a bank customer is likely to churn (i.e., leave the bank). It is built using an **Artificial Neural Network (ANN)** trained on the [Churn_Modelling.csv](https://www.kaggle.com/datasets/shubhendra7/customer-churn-modelling) dataset, and deployed via **Streamlit Cloud**.

👉 **Live Demo**:  

[Click Here](https://mrextinct27-ann-churn-predictor-app-lhjeyk.streamlit.app/)

---

## 🚀 Project Overview

- **Frontend**: Streamlit (interactive UI)
- **Model**: TensorFlow/Keras-based Artificial Neural Network
- **Preprocessing**: Scikit-learn
- **Deployment**: Streamlit Cloud
- **Input Features**:
  - Credit Score
  - Geography (France, Spain, Germany)
  - Gender
  - Age
  - Tenure
  - Balance
  - Number of Products
  - Has Credit Card
  - Is Active Member
  - Estimated Salary

---

## 🧠 Machine Learning Workflow

### 1. **Data Preprocessing**
- Removed irrelevant features (`RowNumber`, `CustomerId`, `Surname`)
- Applied:
  - **Label Encoding** for `Gender`
  - **One-Hot Encoding** for `Geography`
  - **Standard Scaling** for numerical features

### 2. **Model Architecture (ANN)**
- 3 Fully Connected (Dense) layers:
  - Input layer matching number of processed features
  - Hidden layers with ReLU activation
  - Output layer with sigmoid activation for binary classification
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **Evaluation Metric**: Accuracy

### 3. **Callbacks**
- Early stopping to avoid overfitting
- TensorBoard support for visualization (local use)

---

## 🧪 Prediction Logic

When a user enters new customer data in the UI:
1. Input is encoded and scaled using the saved preprocessors.
2. Processed data is passed into the trained ANN model.
3. Prediction is displayed with churn probability.

---
