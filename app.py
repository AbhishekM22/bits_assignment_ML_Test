import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Mobile Price Classifier", layout="wide")
st.title("📱 Mobile Price Classification App")

# Sidebar
model_option = st.sidebar.selectbox(
    "Select ML Model",
    ("logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost")
)

uploaded_file = st.file_uploader("Upload Test CSV", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    
    if 'price_range' in test_data.columns:
        X_test = test_data.drop('price_range', axis=1)
        y_test = test_data['price_range']

        # Load model
        with open(f'model/{model_option}.pkl', 'rb') as f:
            model = pickle.load(f)

        # Predictions
        y_pred = model.predict(X_test.values)
        
        # Probabilities (Required for AUC Score)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test.values)
            auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr')
        else:
            auc_val = "N/A"

        # --- 1. CALCULATE ALL 6 METRICS ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)

        # --- 2. DISPLAY METRICS IN COLUMNS ---
        st.subheader(f"📊 6 Mandatory Metrics for {model_option}")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("1. Accuracy", f"{acc:.2f}")
        col2.metric("2. AUC Score", f"{auc_val:.2f}" if isinstance(auc_val, float) else auc_val)
        col3.metric("3. Precision", f"{prec:.2f}")
        col4.metric("4. Recall", f"{rec:.2f}")
        col5.metric("5. F1 Score", f"{f1:.2f}")
        col6.metric("6. MCC Score", f"{mcc:.2f}")

        # --- 3. VISUALS ---
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
            
        with c2:
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='RdPu')
            st.pyplot(fig)
            
    else:
        st.error("Error: 'price_range' column missing for evaluation.")