# In [1] Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# In [2] Load Dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# In [3] Features and Target
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
y = df["Gender"]

# In [4] Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In [5] Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# In [6] Model Training
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# In [7] Prediction
y_pred = model.predict(X_test)

# In [8] Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))