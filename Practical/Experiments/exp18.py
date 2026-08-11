import numpy as np
import matplotlib.pyplot as plt

# Two-class data
X = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [6, 6],
    [7, 6],
    [7, 7],
    [8, 7]
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Initialize weights and bias
w = np.zeros(2)
b = 0.0

learning_rate = 0.01
epochs = 1000

# Linear activation function
def linear(z):
    return z


# Training
for epoch in range(epochs):

    z = np.dot(X, w) + b

    output = linear(z)

    error = output - y

    dw = np.dot(X.T, error) / len(X)

    db = np.mean(error)

    w = w - learning_rate * dw

    b = b - learning_rate * db


# Prediction
output = np.dot(X, w) + b

prediction = (output >= 0.5).astype(int)

accuracy = np.mean(prediction == y)

print("Final Weights:")
print(w)

print("\nFinal Bias:")
print(b)

print("\nActual Values:")
print(y)

print("\nPredicted Values:")
print(prediction)

print("\nAccuracy:")
print(accuracy)


# Plot
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")

x_values = np.linspace(0, 9, 100)

y_values = (0.5 - w[0] * x_values - b) / w[1]

plt.plot(x_values, y_values, label="Decision Boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Two-Class Neural Network with Linear Activation")
plt.legend()
plt.grid()
plt.show()
