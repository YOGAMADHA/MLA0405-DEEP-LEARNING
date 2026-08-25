import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate two-class data
np.random.seed(42)

n = 500

X1 = np.random.randn(n // 2, 2) + [-2, 0]
X2 = np.random.randn(n // 2, 2) + [2, 0]

X = np.vstack((X1, X2))

y = np.hstack((
    np.zeros(n // 2),
    np.ones(n // 2)
))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Neural Network with ReLU
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    learning_rate_init=0.01,
    max_iter=2000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("==========================================")
print("22. Neural Network Analysis")
print("Two-Class Data - ReLU Activation")
print("==========================================")
print("Activation Function :", "ReLU")
print("Hidden Neurons      :", 10)
print("Learning Rate       :", 0.01)
print("Accuracy            :", round(accuracy * 100, 2), "%")
print("==========================================")

# Decision boundary
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid)
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(7, 6))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolors="black"
)

plt.title("Two-Class Data - ReLU Neural Network")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
