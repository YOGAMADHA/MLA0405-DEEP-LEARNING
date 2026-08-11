import numpy as np
import matplotlib.pyplot as plt

# Three-class data
X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],

    [5, 1],
    [6, 1],
    [5, 2],
    [6, 2],

    [3, 5],
    [4, 5],
    [3, 6],
    [4, 6]
])

# Class labels
y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])

# One-hot encoding
Y = np.zeros((len(y), 3))

Y[np.arange(len(y)), y] = 1

# Initialize weights
W = np.zeros((2, 3))

b = np.zeros(3)

learning_rate = 0.01
epochs = 2000

# Linear activation
def linear(z):
    return z


# Training
for epoch in range(epochs):

    output = linear(np.dot(X, W) + b)

    error = output - Y

    dW = np.dot(X.T, error) / len(X)

    db = np.mean(error, axis=0)

    W = W - learning_rate * dW

    b = b - learning_rate * db


# Prediction
output = np.dot(X, W) + b

prediction = np.argmax(output, axis=1)

accuracy = np.mean(prediction == y)

print("Actual Classes:")
print(y)

print("\nPredicted Classes:")
print(prediction)

print("\nAccuracy:")
print(accuracy)


# Plot
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)

plt.scatter(
    X[y == 2, 0],
    X[y == 2, 1],
    label="Class 2"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Multi-Class Neural Network with Linear Activation")

plt.legend()
plt.grid()

plt.show()
