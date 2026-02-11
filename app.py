import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Mobile Price Classifier")
st.title("📱 Mobile Price Classification App")

# 1. Model Selection
model_option = st.sidebar.selectbox(
    "Select ML Model",
    ("logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost")
)

# 2. Dataset Upload
uploaded_file = st.file_uploader("Upload Test CSV", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", test_data.head())

    if 'price_range' in test_data.columns:
        X_test = test_data.drop('price_range', axis=1)
        y_test = test_data['price_range']

        # Load chosen model
        with open(f'model/{model_option}.pkl', 'rb') as f:
            model = pickle.load(f)

        # Predictions
        y_pred = model.predict(X_test)

        # 3. Display Metrics
        st.subheader(f"Evaluation Metrics: {model_option}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # 4. Confusion Matrix
        st.subheader("Confusion Matrix Visual")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)
    else:
        st.error("Error: 'price_range' column missing for evaluation.")