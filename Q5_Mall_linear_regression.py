# ================= QUESTION 5 =================
# Perform Linear Regression to predict.
# Evaluate the model (R2 etc.) and plot Actual vs Predicted.
# Dataset: Mall_Customers.CSV
# =================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
# Target: Spending Score (1-100); Features: Gender, Age, Annual Income
df = pd.read_csv("Mall_Customers.csv")
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
X = df[["Gender", "Age", "Annual Income (k$)"]]
y = df["Spending Score (1-100)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Coefficients:", model.coef_, "Intercept:", model.intercept_)
evaluate("Mall Customers - Linear Regression", y_test, y_pred)
plot_actual_vs_pred(y_test, y_pred, "Spending Score: Actual vs Predicted")
# Note: is dataset me features aur spending score ka linear relation weak hai, isliye R2 low aayega (normal hai).
