# ================= QUESTION 7 =================
# Decision Tree (drugs)
# Dataset: drug.csv
# =================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# ---- Solution ----
df = pd.read_csv("drug.csv")
print(df.head())

# Categorical columns ko numbers me convert karo
df["Sex"] = df["Sex"].map({"F": 0, "M": 1})
df["BP"] = df["BP"].map({"LOW": 0, "NORMAL": 1, "HIGH": 2})
df["Cholesterol"] = df["Cholesterol"].map({"NORMAL": 0, "HIGH": 1})

X = df[["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]]
y = df["Drug"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=3).fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=tree.classes_), display_labels=tree.classes_).plot(cmap="Blues")
plt.show()

plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X.columns, class_names=tree.classes_, filled=True)
plt.show()
