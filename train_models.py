import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Load Dataset
df = pd.read_csv('train.csv')
X = df.drop('price_range', axis=1)
y = df['price_range']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary of 6 Models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Training Loop
for name, model in models.items():
    model.fit(X_train, y_train)
    # Save each model as a .pkl file in the model folder
    with open(f'model/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Successfully saved: {name}.pkl")