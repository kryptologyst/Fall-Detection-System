#!/usr/bin/env python3
"""Simple Fall Detection Demo - Original Implementation

This is the original simple implementation of the fall detection system.
For the full modernized version, see the src/ directory and use the training scripts.

DISCLAIMER: This is a research demonstration only - NOT FOR CLINICAL USE
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("=" * 60)
print("FALL DETECTION SYSTEM - SIMPLE DEMO")
print("=" * 60)
print("DISCLAIMER: This is a research demonstration only - NOT FOR CLINICAL USE")
print("Always consult with qualified healthcare professionals for medical decisions")
print("=" * 60)

# 1. Simulated sensor dataset: [x, y, z, magnitude], label (1=fall, 0=normal)
np.random.seed(42)
samples = 500
data = {
    "x": np.random.normal(0, 1, samples),
    "y": np.random.normal(0, 1, samples),
    "z": np.random.normal(9.8, 1, samples),
}
df = pd.DataFrame(data)
df["magnitude"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
df["label"] = (df["magnitude"] > 15).astype(int)  # fall if sudden spike in acceleration

print(f"\nDataset created with {samples} samples")
print(f"Fall events: {df['label'].sum()} ({df['label'].mean():.1%})")
print(f"Normal activities: {(df['label'] == 0).sum()} ({(1-df['label'].mean()):.1%})")

# 2. Features and labels
X = df[["x", "y", "z", "magnitude"]]
y = df["label"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# 4. Train classifier
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("\nFall Detection Report:")
print("-" * 40)
print(classification_report(y_test, y_pred, target_names=["Normal", "Fall"]))

# 6. Predict on new event
print("\nTesting on new sensor readings:")
new_event = pd.DataFrame([{"x": 2.3, "y": 1.7, "z": 19.5}])
new_event["magnitude"] = np.sqrt(new_event["x"]**2 + new_event["y"]**2 + new_event["z"]**2)
prediction = model.predict(new_event)[0]
probability = model.predict_proba(new_event)[0]

print(f"Sensor readings: x={new_event['x'].iloc[0]:.1f}, y={new_event['y'].iloc[0]:.1f}, z={new_event['z'].iloc[0]:.1f}")
print(f"Motion magnitude: {new_event['magnitude'].iloc[0]:.1f}")
print(f"Prediction: {'FALL' if prediction == 1 else 'NORMAL'}")
print(f"Confidence: Normal={probability[0]:.3f}, Fall={probability[1]:.3f}")

print("\n" + "=" * 60)
print("For the full modernized system with advanced models,")
print("interactive demo, and comprehensive evaluation, see:")
print("- src/ directory for the complete implementation")
print("- demo/app.py for the Streamlit interactive demo")
print("- scripts/train.py for training advanced models")
print("- scripts/evaluate.py for comprehensive evaluation")
print("=" * 60)