```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

data = pd.read_csv("healthcare_dataset.csv")

print("\nDataset Loaded Successfully")
print("Dataset Shape:", data.shape)
print("\nFirst 5 Records:")
print(data.head())

# ------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------

data = data.drop_duplicates()
data = data.dropna()

print("\nAfter Preprocessing:")
print("Dataset Shape:", data.shape)

# ------------------------------------------------
# SEPARATE FEATURES AND TARGET
# ------------------------------------------------

X = data.drop("Disease", axis=1)
y = data["Disease"]

# Convert categorical input columns
X = pd.get_dummies(X)

# Convert disease names to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------------------------------------
# SPLIT DATA
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nTraining Samples:", len(X_train))
print("Validation Samples:", len(X_val))
print("Testing Samples:", len(X_test))

# ------------------------------------------------
# CREATE DEEP FEED-FORWARD ANN
# ------------------------------------------------

model = Sequential()

model.add(
    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

model.add(Dropout(0.30))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(Dropout(0.20))

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        len(np.unique(y)),
        activation="softmax"
    )
)

# ------------------------------------------------
# COMPILE MODEL
# ------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nDeep Feed-Forward ANN Model:")
model.summary()

# ------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

# ------------------------------------------------
# TEST MODEL
# ------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

probabilities = model.predict(X_test)

y_pred = np.argmax(probabilities, axis=1)

# ------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)

# ------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

# ------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ------------------------------------------------
# ACCURACY GRAPH
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

# ------------------------------------------------
# LOSS GRAPH
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()

# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------

model.save("healthcare_disease_model.keras")

print("\nModel saved successfully!")
print("File: healthcare_disease_model.keras")
```
