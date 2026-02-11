# Mobile Price Classification - BITS Assignment 2

## a. Problem Statement
The goal of this project is to classify the price range of mobile phones based on their hardware specifications (RAM, Battery Power, Internal Memory, etc.). This is a multi-class classification problem with four price ranges: 0 (low cost), 1 (medium cost), 2 (high cost), and 3 (very high cost).

## b. Dataset Description
- **Source:** Mobile Price Classification dataset (Kaggle/UCI)
- **Instances:** 2000
- **Features:** 20 (including RAM, Battery Power, Pixel Resolution, etc.)
- **Target Variable:** `price_range` (0, 1, 2, 3)

## c. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.96 | 0.99 | 0.96 | 0.96 | 0.96 | 0.94 |
| Decision Tree | 0.83 | 0.89 | 0.83 | 0.83 | 0.83 | 0.77 |
| kNN | 0.92 | 0.97 | 0.92 | 0.92 | 0.92 | 0.89 |
| Naive Bayes | 0.80 | 0.95 | 0.80 | 0.80 | 0.79 | 0.73 |
| Random Forest | 0.88 | 0.98 | 0.88 | 0.88 | 0.88 | 0.84 |
| XGBoost | 0.90 | 0.99 | 0.90 | 0.90 | 0.90 | 0.87 |

## d. Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Highest accuracy; the relationship between specs like RAM and Price is highly linear. |
| Decision Tree | Faster training but lower accuracy than ensembles; prone to slight overfitting. |
| kNN | Strong performance; mobile phones with similar specs naturally fall into the same price clusters. |
| Naive Bayes | Lowest performance; likely due to the assumption that all features (like 3G/4G) are independent. |
| Random Forest | Very robust; handles the 20 features well without much tuning. |
| XGBoost | Excellent performance and high AUC score; very reliable for this tabular data. |