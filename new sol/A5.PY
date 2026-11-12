# In [1] Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# In [2] Load Dataset
df = pd.read_csv("/kaggle/input/mall-customers/Mall_Customers.csv")
print(df.head())

# In [3] Features and Target
X = df[["Age", "Annual Income (k$)"]]
y = df["Spending Score (1-100)"]

# In [4] Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In [5] Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# In [6] Prediction
y_pred = model.predict(X_test)

# In [7] Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

# In [8] Actual vs Predicted Plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual Spending Score")
plt.ylabel("Predicted Spending Score")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()