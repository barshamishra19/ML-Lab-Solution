# ============================================================
# Assignment 03: Linear Regression with L1 (Lasso) and L2 (Ridge)
# Predict House Price | Dataset: housing_price_dataset.csv
# ============================================================

# In [1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# In [2]  Load dataset
# NOTE: path apne dataset ke hisaab se badal lena
df = pd.read_csv("/kaggle/input/datasets/yomnamadan/usa-housing/USA_Housing.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# In [3]  Features (X) and Target (y)
X = df.drop(columns=["Price", "Address"])
y = df["Price"]

# In [4]  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# In [5]  Hyperparameter range (alpha values)
alphas = np.logspace(-4, 4, 30)

# ------------------------------------------------------------
# LASSO (L1) with GridSearchCV
# ------------------------------------------------------------
# In [6]
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=100000, random_state=42))
])

lasso_grid = GridSearchCV(
    lasso_pipeline,
    param_grid={"model__alpha": alphas},
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# In [7]
lasso_grid.fit(X_train, y_train)
lasso_best_alpha = lasso_grid.best_params_["model__alpha"]
lasso_pred = lasso_grid.predict(X_test)

lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_test, lasso_pred)

print("\n======== LASSO REGRESSION ========")
print("Best Alpha:", lasso_best_alpha)
print("MAE       :", lasso_mae)
print("MSE       :", lasso_mse)
print("RMSE      :", lasso_rmse)
print("R² Score  :", lasso_r2)
print("CV R²     :", lasso_grid.best_score_)

# ------------------------------------------------------------
# RIDGE (L2) with GridSearchCV
# ------------------------------------------------------------
# In [8]
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

ridge_grid = GridSearchCV(
    ridge_pipeline,
    param_grid={"model__alpha": alphas},
    cv=5,
    scoring="r2",
    n_jobs=-1
)

ridge_grid.fit(X_train, y_train)
ridge_best_alpha = ridge_grid.best_params_["model__alpha"]
ridge_pred = ridge_grid.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_pred)

print("\n======== RIDGE REGRESSION ========")
print("Best Alpha:", ridge_best_alpha)
print("MAE       :", ridge_mae)
print("MSE       :", ridge_mse)
print("RMSE      :", ridge_rmse)
print("R² Score  :", ridge_r2)
print("CV R²     :", ridge_grid.best_score_)

# In [9]  Model comparison table
results = pd.DataFrame({
    "Model": ["Lasso (L1)", "Ridge (L2)"],
    "Best Alpha": [
        lasso_best_alpha,
        ridge_best_alpha
    ],
    "MAE": [
        lasso_mae,
        ridge_mae
    ],
    "MSE": [
        lasso_mse,
        ridge_mse
    ],
    "RMSE": [
        lasso_rmse,
        ridge_rmse
    ],
    "R2": [
        lasso_r2,
        ridge_r2
    ]
})
print("\n======== MODEL COMPARISON ========")
print(results)

# ------------------------------------------------------------
# Actual vs Predicted plots
# ------------------------------------------------------------
# In [10]  Lasso
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    lasso_pred,
    alpha=0.6
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted - Lasso (L1)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# In [11]  Ridge
plt.figure(figsize=(8, 6))
plt.scatter(
    y_test,
    ridge_pred,
    alpha=0.6
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted - Ridge (L2)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
