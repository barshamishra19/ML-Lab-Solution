# In [1] Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# In [2] Load Dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# In [3] Features and Target
X = pd.get_dummies(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
y = df["Gender"]

# In [4] Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In [5] Model Training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# In [6] Prediction
y_pred = model.predict(X_test)

# In [7] Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))