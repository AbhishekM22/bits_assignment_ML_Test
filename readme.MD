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
| Logistic Regression | 0.72 | 0.92 | 0.72 | 0.72 | 0.72 | 0.63 |
| Decision Tree | 0.96 | 0.98 | 0.96 | 0.96 | 0.96 | 0.95 |
| kNN | 0.95 | 1.00 | 0.95 | 0.95 | 0.95 | 0.93 |
| Naive Bayes | 0.82 | 0.96 | 0.82 | 0.82 | 0.82| 0.76 |
| Random Forest | 0.97 | 1.00 | 0.97 | 0.97 | 0.97 | 0.96 |
| XGBoost | 0.98 | 1.00 | 0.98 | 0.98 | 0.98 | 0.97 |

## d. Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved the lowest accuracy (0.72). This suggests the relationship between features and price categories is non-linear, making a simple linear classifier less effective than tree-based models. |
| Decision Tree | Strong performance with 0.96 accuracy. It effectively captured the decision boundaries of the dataset, though it is slightly outperformed by ensemble methods that reduce variance. |
| kNN | Excellent performance (0.95 accuracy) and a perfect AUC of 1.00. This indicates that mobile phones with similar hardware specifications cluster very tightly within their respective price ranges.. |
| Naive Bayes | Moderate performance (0.82). While computationally fast, its assumption of feature independence likely hindered it since features like RAM and Battery are often correlated. |
| Random Forest | Outstanding results with 0.97 accuracy. By averaging multiple decision trees, it achieved high stability and a perfect AUC score, handling the 20 features without overfitting. |
| XGBoost | The best performing model (0.98 accuracy). Its gradient boosting approach minimized errors more effectively than any other model, making it the most reliable for this classification task. |