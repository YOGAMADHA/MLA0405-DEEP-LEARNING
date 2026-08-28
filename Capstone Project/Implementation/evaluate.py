import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# SETTINGS
# =========================

DATASET_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# =========================
# LOAD VALIDATION DATA
# =========================

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False
)

class_names = test_ds.class_names

print("Classes:", class_names)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "signature_forgery_model.keras"
)

# =========================
# PREDICTIONS
# =========================

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = (
        predictions.flatten() >= 0.5
    ).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\n================================")
print("       CONFUSION MATRIX")
print("================================")

print(cm)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    y_true,
    y_pred
)

# =========================
# PRECISION
# =========================

precision = precision_score(
    y_true,
    y_pred,
    labels=[0, 1],
    average="macro",
    zero_division=0
)

# =========================
# RECALL
# =========================

recall = recall_score(
    y_true,
    y_pred,
    labels=[0, 1],
    average="macro",
    zero_division=0
)

# =========================
# F1 SCORE
# =========================

f1 = f1_score(
    y_true,
    y_pred,
    labels=[0, 1],
    average="macro",
    zero_division=0
)

# =========================
# DISPLAY METRICS
# =========================

print("\n================================")
print("       MODEL PERFORMANCE")
print("================================")

print(
    "Accuracy  :",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision :",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall    :",
    round(recall * 100, 2),
    "%"
)

print(
    "F1-Score  :",
    round(f1 * 100, 2),
    "%"
)

print("================================")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n================================")
print("       CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        zero_division=0
    )
)

# =========================
# CONFUSION MATRIX GRAPH
# =========================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title(
    "Signature Forgery Detection"
)

plt.savefig(
    "confusion_matrix.png"
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")