# ================= QUESTION 2 =================
# Perform Linear Regression with L1 (Lasso) and L2 (Ridge) regularization to predict the price of a House.
# Use hyper-parameter tuning for the best result. Evaluate accuracy (R2 etc.)
# and plot Actual vs Predicted for both models.
# Dataset: housing_price_dataset.csv
# =================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_xy(path, target_hint):
    """CSV load -> (X, y). Target column ko naam se dhundhta hai, categorical ko one-hot karta hai."""
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates()
    cols = [c for c in df.columns if target_hint.lower() in c.lower()]
    target = cols[0] if cols else df.columns[-1]          # na mile to last column
    print("Target column:", target)
    y = df[target]
    X = df.drop(columns=[target])
    # bahut zyada unique values wale text columns (jaise name/id) hata do
    X = X.drop(columns=[c for c in X.select_dtypes("object") if X[c].nunique() > 20])
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def evaluate(name, y_test, y_pred):
    print(f"--- {name} ---")
    print("R2 Score :", round(r2_score(y_test, y_pred), 4))
    print("MAE      :", round(mean_absolute_error(y_test, y_pred), 4))
    print("MSE      :", round(mean_squared_error(y_test, y_pred), 4))
    print("RMSE     :", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))

def plot_actual_vs_pred(y_test, y_pred, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.show()

# ---- Solution ----
X, y = load_xy("housing_price_dataset.csv", target_hint="price")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

for name, reg in [("Lasso (L1)", Lasso(max_iter=50000)), ("Ridge (L2)", Ridge())]:
    pipe = Pipeline([("scaler", StandardScaler()), ("model", reg)])
    grid = GridSearchCV(pipe, params, cv=5, scoring="r2")        # hyper-parameter tuning
    grid.fit(X_train, y_train)
    print(f"\nBest alpha for {name}:", grid.best_params_["model__alpha"])
    y_pred = grid.predict(X_test)
    evaluate(name, y_test, y_pred)
    plot_actual_vs_pred(y_test, y_pred, f"House Price - {name}: Actual vs Predicted")
