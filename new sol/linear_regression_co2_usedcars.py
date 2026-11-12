# ============================================================
# Assignment 01: Linear Regression
# a) CO2 Emission      (fuel_consumption_dataset.csv)
# b) Used Car Price    (used_cars_dataset.csv)
# ============================================================

# In [1]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PART (a): CO2 EMISSION
# ============================================================

# In [2]  Load dataset
df = pd.read_csv("/kaggle/input/datasets/puspitamaji/fuel-consumption-dataset/fuel-consumption-dataset (1).csv")
print(df.head())

# In [3]  Target (Y) and Feature (X)
Y = df["CO2 EMISSIONS"]
X = df[["ENGINESIZE"]]
print("Extracted Feature:")
print(X.head())
print(Y.head())

# In [4]  Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# In [5]  Model training
model = LinearRegression()
model.fit(X_train, Y_train)

# In [6]  Prediction
Y_pred = model.predict(X_test)

# In [7]  Evaluation
print("MAE      :", mean_absolute_error(Y_test, Y_pred))
print("MSE      :", mean_squared_error(Y_test, Y_pred))
print("RMSE     :", np.sqrt(mean_squared_error(Y_test, Y_pred)))
print("R2 Score :", r2_score(Y_test, Y_pred))

# In [8]  Actual vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(Y_test, Y_pred)
min_val = min(Y_test.min(), Y_pred.min())
max_val = max(Y_test.max(), Y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted CO2 Emission")
plt.grid(True)
plt.show()


# ============================================================
# PART (b): USED CAR SELLING PRICE
# ============================================================

# In [9]  Load dataset
# NOTE: apne dataset ke hisaab se path aur target column name check kar lena
df2 = pd.read_csv("/kaggle/input/datasets/puspitamaji/used-cars-dataset/used_cars_dataset.csv")
print("Dataset Shape:", df2.shape)
print(df2.head())
print(df2.info())

# In [10]  Missing values hatao
df2 = df2.dropna()

# In [11]  Target (y) and Features (X)
# Target column ka naam dataset ke hisaab se badlo (e.g. "selling_price" / "Price")
target_col = "selling_price"
y = df2[target_col]
X = df2.drop(target_col, axis=1)

# Numeric aur categorical columns alag karo
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# In [12]  Categorical columns ko encode karo (OneHotEncoder) + Linear Regression
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

model2 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# In [13]  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In [14]  Model training
model2.fit(X_train, y_train)

# In [15]  Prediction
y_pred = model2.predict(X_test)

# In [16]  Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE      :", mae)
print("MSE      :", mse)
print("RMSE     :", rmse)
print("R2 Score :", r2)

# In [17]  Actual vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Used Car Price")
plt.grid(True)
plt.show()
