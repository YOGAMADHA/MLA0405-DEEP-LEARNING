```python
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load trained model
model = load_model("healthcare_disease_model.keras")

# Load dataset to reproduce preprocessing
data = pd.read_csv("healthcare_dataset.csv")

data = data.drop_duplicates()
data = data.dropna()

X = data.drop("Disease", axis=1)
y = data["Disease"]

# Convert categorical columns
X = pd.get_dummies(X)

# Encode disease names
label_encoder = LabelEncoder()
label_encoder.fit(y)

# Scale features
scaler = StandardScaler()
scaler.fit(X)

# Select one sample from the dataset for demonstration
sample = X.iloc[[0]]

sample = scaler.transform(sample)

# Predict
prediction = model.predict(sample)

predicted_class = np.argmax(prediction)

disease = label_encoder.inverse_transform(
    [predicted_class]
)[0]

confidence = np.max(prediction) * 100

print("\n==============================")
print("HEALTHCARE DISEASE PREDICTION")
print("==============================")

print("Predicted Disease:", disease)
print("Confidence:", round(confidence, 2), "%")
```
