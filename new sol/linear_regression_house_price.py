# ============================================================
# Assignment 02: Linear Regression
# Predict the selling price of a house | Dataset: house_price_dataset.csv
# ============================================================

# In [1]  Load dataset, target (y) and features (X)
# NOTE: path apne dataset ke hisaab se badal lena
import pandas as pd
df = pd.read_csv("/kaggle/input/datasets/puspitamaji/housing/price-dataset (1).csv")
y = df['Price']
X = df.drop('Price', axis=1)
print("Extract Feature:")
print(X.columns)

# In [2]  Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# In [3]  Numeric aur categorical columns alag karo
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# In [4]  Preprocessing (categorical columns ko OneHotEncode karo)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# In [5]  Pipeline: preprocessing + Linear Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# In [6]  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In [7]  Model training
model.fit(X_train, y_train)

# In [8]  Prediction
y_pred = model.predict(X_test)

# In [9]  Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# In [10]  Actual vs Predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.show()
