"""
3_train_fraud_detector.py
Freelancer fraud detection with realistic dataset, IsolationForest + RandomForest.
"""

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score
)
import joblib

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = '../models/fraud'
DATASET_PATH = '../datasets/fraud_dataset.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODULE 4: Fraud Detection – Training")
print("=" * 60)

# Generate realistic freelancing fraud dataset
print("\n[STEP 1] Generating fraud dataset (5000 rows, 95/5 split)...")

def gen_normal():
    return {
        'apps_per_hour': np.random.poisson(2) + random.uniform(0.5, 3),
        'geo_mismatch': random.choices([0, 1], weights=[95, 5])[0],
        'bio_keyword_suspicion': random.uniform(0, 0.3),
        'profile_age_days': random.randint(30, 2000),
        'identical_proposals': random.choices([0, 1, 2], weights=[85, 10, 5])[0],
        'payment_dispute_rate': random.uniform(0, 0.05),
        'skill_verification_failed': random.choices([0, 1], weights=[90, 10])[0],
        'avg_response_time_hours': random.uniform(0.5, 12),
        'label': 0
    }

def gen_fraud():
    return {
        'apps_per_hour': random.uniform(15, 50),
        'geo_mismatch': random.choices([0, 1], weights=[10, 90])[0],
        'bio_keyword_suspicion': random.uniform(0.6, 1.0),
        'profile_age_days': random.randint(0, 20),
        'identical_proposals': random.randint(5, 20),
        'payment_dispute_rate': random.uniform(0.3, 1.0),
        'skill_verification_failed': random.choices([0, 1], weights=[10, 90])[0],
        'avg_response_time_hours': random.uniform(48, 200),
        'label': 1
    }

total = 5000
n_fraud = int(total * 0.05)
n_normal = total - n_fraud

rows = [gen_normal() for _ in range(n_normal)] + [gen_fraud() for _ in range(n_fraud)]
random.shuffle(rows)
df = pd.DataFrame(rows)
df.to_csv(DATASET_PATH, index=False)
print(f"[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Class distribution:\n{df['label'].value_counts()}")

feature_cols = [
    'apps_per_hour', 'geo_mismatch', 'bio_keyword_suspicion',
    'profile_age_days', 'identical_proposals', 'payment_dispute_rate',
    'skill_verification_failed', 'avg_response_time_hours'
]
X = df[feature_cols].values
y = df['label'].values

# ─────────────── Model A: RandomForest (Primary) ───────────────
print("\n[STEP 2] Training RandomForestClassifier (primary)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1))
])

cv_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
print(f"  CV F1 Scores: {np.round(cv_scores, 4)} | Mean: {cv_scores.mean():.4f}")

# GridSearchCV
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10],
    'clf__min_samples_split': [2, 5]
}
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
grid.fit(X, y)
best_rf = grid.best_estimator_
print(f"  Best params: {grid.best_params_}")

y_pred = best_rf.predict(X)
y_proba = best_rf.predict_proba(X)[:, 1]
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc = roc_auc_score(y, y_proba)

print(f"\n[STEP 3] RandomForest Metrics:")
print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
print(f"\n[Classification Report]\n{classification_report(y, y_pred)}")
print(f"[Confusion Matrix]\n{confusion_matrix(y, y_pred)}")

# Feature importances
print("\n[Feature Importance]")
importances = best_rf.named_steps['clf'].feature_importances_
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# Save versioned model
f1_pct = int(f1 * 100)
model_filename = f"fraud_detector_rf_f1{f1_pct}.pkl"
model_path = os.path.join(OUTPUT_DIR, model_filename)
canonical_path = os.path.join(OUTPUT_DIR, 'fraud_detector_rf.pkl')
joblib.dump(best_rf, model_path)
joblib.dump(best_rf, canonical_path)
print(f"\n[OK] RF saved: {canonical_path}")

# ─────────────── Model B: IsolationForest (Backup) ───────────────
print("\n[STEP 4] Training IsolationForest (backup anomaly detector)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(contamination=0.05, random_state=RANDOM_SEED, n_jobs=-1)
iso.fit(X_scaled)
iso_pred = iso.predict(X_scaled)
iso_labels = np.where(iso_pred == -1, 1, 0)

iso_prec = precision_score(y, iso_labels, zero_division=0)
iso_rec = recall_score(y, iso_labels, zero_division=0)
iso_f1 = f1_score(y, iso_labels, zero_division=0)
print(f"  IsolationForest -> Precision: {iso_prec:.4f} | Recall: {iso_rec:.4f} | F1: {iso_f1:.4f}")

# Save IsolationForest with scaler
iso_bundle = {'model': iso, 'scaler': scaler}
iso_path = os.path.join(OUTPUT_DIR, 'fraud_detector_iso.pkl')
joblib.dump(iso_bundle, iso_path)
print(f"[OK] IsolationForest bundle saved: {iso_path}")

# Save metadata
metadata = {
    'module': 'Fraud Detection',
    'primary_model': 'RandomForest',
    'backup_model': 'IsolationForest',
    'cv_f1_scores': cv_scores.tolist(),
    'mean_f1': float(cv_scores.mean()),
    'final_accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1_score': float(f1),
    'roc_auc': float(roc),
    'isolation_forest_f1': float(iso_f1),
    'best_params': grid.best_params_,
    'feature_cols': feature_cols,
    'model_file': model_filename,
    'timestamp': datetime.now().isoformat()
}
with open(os.path.join(OUTPUT_DIR, 'model_report.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

GLOBAL_METRICS = '../models/training_metrics.json'
gm = json.load(open(GLOBAL_METRICS)) if os.path.exists(GLOBAL_METRICS) else {}
gm['Fraud Detection'] = metadata
with open(GLOBAL_METRICS, 'w') as f:
    json.dump(gm, f, indent=4)

print("\n[OK] MODULE 4 TRAINING COMPLETE")
