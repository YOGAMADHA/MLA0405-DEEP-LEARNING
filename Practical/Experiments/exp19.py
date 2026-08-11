import numpy as np
import matplotlib.pyplot as plt

# Generate circular data
np.random.seed(42)

n = 100

# Inner circle - Class 0
angles1 = np.random.rand(n) * 2 * np.pi
radius1 = np.random.rand(n) * 2

x1 = radius1 * np.cos(angles1)
y1 = radius1 * np.sin(angles1)

# Outer circle - Class 1
angles2 = np.random.rand(n) * 2 * np.pi
radius2 = 3 + np.random.rand(n) * 2

x2 = radius2 * np.cos(angles2)
y2 = radius2 * np.sin(angles2)

# Combine data
X = np.vstack((
    np.column_stack((x1, y1)),
    np.column_stack((x2, y2))
))

y = np.array([0] * n + [1] * n)

# Initialize weights
w = np.zeros(2)
b = 0.0

learning_rate = 0.01
epochs = 1000

# Linear activation
def linear(z):
    return z


# Training
for epoch in range(epochs):

    output = np.dot(X, w) + b

    error = output - y

    dw = np.dot(X.T, error) / len(X)

    db = np.mean(error)

    w = w - learning_rate * dw

    b = b - learning_rate * db


# Prediction
output = np.dot(X, w) + b

prediction = (output >= 0.5).astype(int)

accuracy = np.mean(prediction == y)

print("Accuracy:")
print(accuracy)

print("\nResult:")
print("Linear activation cannot correctly separate circular data.")

# Plot
plt.scatter(x1, y1, label="Class 0")
plt.scatter(x2, y2, label="Class 1")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Circular Data with Linear Activation")

plt.legend()
plt.grid()

plt.axis("equal")

plt.show()
