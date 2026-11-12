# ================= QUESTION 4 =================
# Perform Logistic Regression to classify if a patient has a benign or malignant tumor.
# Generate confusion matrix; compute Precision, Recall, Accuracy, F1-Score etc.
# Plot the ROC curve and calculate AUC.
# Dataset: samples_cancer.csv
# =================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score)

# ---- Solution ----
df = pd.read_csv("samples_cancer.csv")
df["BareNuc"] = pd.to_numeric(df["BareNuc"], errors="coerce")   # '?' values ko NaN banao
df = df.dropna().drop(columns=["ID"])
y = (df["Class"] == 4).astype(int)        # 2 = benign (0), 4 = malignant (1)
X = df.drop(columns=["Class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)
sc = StandardScaler().fit(X_train)
X_train, X_test = sc.transform(X_train), sc.transform(X_test)

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(cmap="Blues")
plt.title("Confusion Matrix"); plt.show()

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(); plt.show()
print("AUC:", auc)
