import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate spiral data
np.random.seed(42)

n = 300

theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

r = 2 * theta + np.pi

x1 = r * np.cos(theta)
y1 = r * np.sin(theta)

x2 = -r * np.cos(theta)
y2 = -r * np.sin(theta)

X1 = np.column_stack((x1, y1))
X2 = np.column_stack((x2, y2))

X = np.vstack((X1, X2))

y = np.hstack((
    np.zeros(n),
    np.ones(n)
))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Neural Network with Sigmoid activation
model = MLPClassifier(
    hidden_layer_sizes=(20, 20),
    activation="logistic",
    learning_rate_init=0.01,
    max_iter=3000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("==========================================")
print("23. Neural Network Analysis")
print("Spiral Data - Sigmoid Activation")
print("==========================================")
print("Activation Function :", "Sigmoid")
print("Hidden Layers       :", "20, 20")
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
plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolors="black"
)

plt.title("Spiral Data - Sigmoid Neural Network")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
