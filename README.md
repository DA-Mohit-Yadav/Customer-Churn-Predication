# Telco Customer Churn Prediction

![Project Banner](https://placehold.co/1200x300/6366f1/FFFFFF?text=Telco+Churn+Prediction)

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results and Evaluation](#-results-and-evaluation)
- [How to Use](#-how-to-use)
- [Model Serialization](#-model-serialization)
- [Future Improvements](#-future-improvements)

---

## 🚀 Project Overview

This project focuses on building a machine learning model to predict customer churn for a telecommunications company. By analyzing a historical dataset of customer attributes and behaviors, the model identifies key factors that influence a customer's decision to leave the service. The ultimate goal is to create a tool that can flag at-risk customers, allowing the company to take proactive measures to improve customer retention.

The entire workflow, from data cleaning and exploratory analysis to model training and evaluation, is documented in the accompanying Jupyter Notebook (`Customer Churn.ipynb`).

---

## 🎯 Problem Statement

Customer churn is a critical challenge for subscription-based businesses. The primary objective of this project is to develop a classification model that accurately predicts which customers are likely to churn. This predictive capability enables the business to:

* Proactively engage with at-risk customers through targeted marketing campaigns or special offers.
* Understand the key drivers of churn to improve products and services.
* Reduce revenue loss by improving customer retention rates.

---

## 📊 Dataset

The project utilizes the **Telco Customer Churn** dataset, sourced from Kaggle. Each row represents a unique customer, and the columns contain various attributes.

* **Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Size:** 7043 customer records
* **Features Include:**
    * **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
    * **Account Information:** `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`
    * **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
    * **Target Variable:** `Churn` (Yes/No)

---

## 🛠️ Methodology

The project follows a structured machine learning workflow:

### 1. Data Cleaning and Preprocessing

* **Handled Missing Values:** Identified and handled missing data in the `TotalCharges` column, which was initially stored as an object type due to empty strings.
* **Corrected Data Types:** Converted `TotalCharges` from `object` to `float`.
* **Removed Multicollinearity:** Dropped the `TotalCharges` column due to its high correlation with `tenure`, and removed the redundant `customerID` column.
* **Feature Engineering:** Simplified categorical features by replacing values like "No internet service" and "No phone service" with a simple "No", making the features more consistent for modeling.

### 2. Exploratory Data Analysis (EDA)

* Visualized the distribution of categorical and numerical features to understand their relationship with the target variable, `Churn`.
* **Key Insights Uncovered:**
    * Customers with **shorter tenure** (especially < 5 months) have a significantly higher churn rate.
    * **Month-to-month contracts** are a major predictor of churn compared to one-year or two-year contracts.
    * Customers with **Fiber optic** internet service are more likely to churn than those with DSL.
    * Lack of services like **Online Security** and **Tech Support** correlates with higher churn.
    * **Senior Citizens** have a higher churn rate than younger customers.

### 3. Model Building

* **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets.
* **Encoding:** `OneHotEncoder` was used for multi-class categorical features (`Contract`, `InternetService`, `PaymentMethod`), while other binary features were label-encoded.
* **Scaling:** `StandardScaler` was applied to numerical features (`tenure`, `MonthlyCharges`) to standardize their scale.
* **Algorithm:** A **Decision Tree Classifier** was chosen as the baseline model.
* **Handling Class Imbalance:** The dataset was found to be imbalanced (73% non-churn vs. 27% churn). To address this, the `class_weight='balanced'` parameter was used during model training to penalize misclassifications of the minority class (churn) more heavily.

---

## 📈 Results and Evaluation

The model's performance was evaluated on the unseen test set. The primary focus was on **Recall** for the churn class, as it is more costly for the business to miss a customer who is about to churn (a False Negative) than to mistakenly flag a happy customer (a False Positive).

| Metric         | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| **Churn (1)** | 0.51      | **0.76** | 0.61     |
| **No Churn (0)**| 0.90      | 0.74   | 0.81     |
| **Accuracy** |           |        | **0.74** |

- **Key Result:** The model successfully identified **76% of all actual churners** in the test set (Recall = 0.76). This provides the business with a strong list of customers to target for retention efforts.

---



---

## 📦 Model Serialization

The final trained Decision Tree model (`dt`) and the preprocessors (`StandardScaler`, `OneHotEncoder`) have been serialized using **pickle**. This allows the trained model to be easily loaded and used for predictions on new data without needing to be retrained, forming the basis for deployment.

---

## 🔮 Future Improvements

* **Advanced Imbalance Handling:** Implement **SMOTE** (Synthetic Minority Over-sampling Technique) on the training data to create synthetic examples of the churn class, which may provide a better balance for the model to learn from.
* **Experiment with Other Models:** Train and evaluate more powerful algorithms like **Random Forest**, **XGBoost**, or **LightGBM**, which often yield higher performance on tabular data.
* **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to systematically find the optimal hyperparameters for the chosen model, which could further boost performance.
