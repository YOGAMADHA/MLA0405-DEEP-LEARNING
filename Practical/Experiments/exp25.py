import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate circular data
np.random.seed(42)

n = 500

angles = np.random.rand(n) * 2 * np.pi
radius = np.random.choice([1, 3], n) + np.random.randn(n) * 0.2

X = np.column_stack((
    radius * np.cos(angles),
    radius * np.sin(angles)
))

# Create two classes
y = (radius > 2).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Neural Network with Tanh activation
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="tanh",
    learning_rate_init=0.01,
    max_iter=2000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("==========================================")
print("25. Neural Network Analysis")
print("Circular Data - Tanh Activation")
print("==========================================")
print("Activation Function :", "Tanh")
print("Hidden Neurons      :", 10)
print("Learning Rate       :", 0.01)
print("Accuracy            :", round(accuracy * 100, 2), "%")
print("==========================================")

# Create decision boundary
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

plt.title("Circular Data - Tanh Neural Network")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
