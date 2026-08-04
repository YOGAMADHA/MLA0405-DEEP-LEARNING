import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

wine = load_wine()

data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
data['Target'] = wine.target
data = data[data['Target'] != 2]

x = data.drop('Target', axis=1)
y = data['Target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1
)

model = DecisionTreeClassifier(random_state=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)

class_report = classification_report(
    y_test, y_pred, target_names=wine.target_names[:2]
)
print("Classification Report:\n", class_report)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='PuBuGn',
    xticklabels=wine.target_names[:2],
    yticklabels=wine.target_names[:2]
)

plt.xlabel("predicted label")
plt.ylabel("true label")
plt.title("confusion matrix")
plt.show()
