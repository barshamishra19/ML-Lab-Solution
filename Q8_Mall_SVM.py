# ================= QUESTION 8 =================
# Using SVM (Mall_Customer.CSV)
# =================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Mall_Customers me koi class column nahi hai, isliye Spending Score se class banayi hai:
# Low (<40), Medium (40-60), High (>60)
df = pd.read_csv("Mall_Customers.csv")
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df["SpendClass"] = pd.cut(df["Spending Score (1-100)"], bins=[0, 40, 60, 100],
                          labels=["Low", "Medium", "High"])
X = df[["Gender", "Age", "Annual Income (k$)"]]
y = df["SpendClass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def report(model_name, y_test, y_pred, labels):
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=labels), display_labels=labels).plot(cmap="Blues")
    plt.title(model_name + " - Confusion Matrix"); plt.show()

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# ---- Solution ----
pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
grid = GridSearchCV(pipe, {"svm__kernel": ["linear", "rbf"], "svm__C": [0.1, 1, 10, 100], "svm__gamma": ["scale", 0.1, 0.01]}, cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
report("SVM", y_test, grid.predict(X_test), ["Low", "Medium", "High"])
