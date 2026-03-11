import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

print("--- TRAINING FRAUD DETECTION ENGINE (MODULE 4) ---")

# 1. Dataset Generation: Abnormal Behavior Patterns (Unsupervised / Anomaly ML)
# We train an Isolation Forest on "normal" behavior to flag "abnormal" behavior.
np.random.seed(42)

# Normal profiles
n_normal = 2000
normal_geo_mismatch = np.random.choice([0], size=n_normal) # No mismatch
normal_apps_per_hour = np.random.poisson(2, size=n_normal) # Normal velocity
normal_bio_suspicion = np.random.uniform(0, 0.1, size=n_normal) # Clean bio

# Fraudulent profiles (Anomalies)
n_anomalies = 100
anom_geo_mismatch = np.random.choice([0, 1], size=n_anomalies, p=[0.2, 0.8])
anom_apps_per_hour = np.random.poisson(35, size=n_anomalies) # Bot speed
anom_bio_suspicion = np.random.uniform(0.5, 1.0, size=n_anomalies) # High suspicion

X_normal = np.column_stack((normal_geo_mismatch, normal_apps_per_hour, normal_bio_suspicion))
X_anomaly = np.column_stack((anom_geo_mismatch, anom_apps_per_hour, anom_bio_suspicion))

# Combine dataset
X = np.vstack((X_normal, X_anomaly))
# Labels for evaluation: 1 for normal, -1 for anomaly (matching Isolation Forest output format)
y_true = np.concatenate((np.ones(n_normal), -np.ones(n_anomalies)))

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y_true = y_true[indices]

# 2. Train Isolation Forest (Unsupervised)
print("Training Isolation Forest Model on behavior footprints...")
# Assume around 5% contamination rate
iso_forest = IsolationForest(contamination=float(n_anomalies)/len(X), random_state=42)
iso_forest.fit(X)

# 3. Detection Evaluation
print("\n--- EVALUATION METRICS ---")
y_pred = iso_forest.predict(X)

# Convert -1/1 labels to logical representation if needed, but we'll use raw for metrics
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Anomaly (-1)", "Normal (1)"]))

# 4. Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(iso_forest, os.path.join("models", "fraud_detector_iso.pkl"))

print("\nModel saved to models/fraud_detector_iso.pkl")
