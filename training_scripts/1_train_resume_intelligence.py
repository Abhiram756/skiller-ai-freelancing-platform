"""
1_train_resume_intelligence.py
Multi-model Resume classifier with StratifiedKFold + GridSearchCV + versioning.
"""

import os
import sys
import json
import subprocess
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

warnings.filterwarnings('ignore')

OUTPUT_DIR = '../models/resume'
DATASET_PATH = '../datasets/resume_dataset.csv'
METRICS_PATH = os.path.join(OUTPUT_DIR, 'model_report.json')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODULE 1: Resume Intelligence – Training")
print("=" * 60)

# Step 1: Prepare dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("\n[STEP 1] Preparing dataset...")
subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "0_prepare_resume_data.py")], check=True)

# Step 2: Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Categories: {df['Category'].unique()}")

# Step 3: Generate embeddings
print("\n[STEP 2] Generating SentenceTransformer embeddings (all-MiniLM-L6-v2)...")
s_model = SentenceTransformer('all-MiniLM-L6-v2')
X = s_model.encode(df['Resume'].tolist(), show_progress_bar=True)
y = df['Category'].values
print(f"[INFO] Embedding shape: {X.shape}")

# Step 4: Define models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

candidates = {
    'RandomForest': Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1))
    ]),
    'SVM_RBF': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ]),
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])
}

# Step 5: Cross-validate all models
print("\n[STEP 3] Cross-validating all models (StratifiedKFold, 5 folds)...")
cv_results = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name}: mean={scores.mean():.4f} std={scores.std():.4f} | scores={np.round(scores, 4)}")

# Step 6: Pick best model
best_name = max(cv_results, key=lambda k: cv_results[k].mean())
print(f"\n[INFO] Best model: {best_name} (mean CV accuracy={cv_results[best_name].mean():.4f})")

# Step 7: GridSearchCV on best model
print(f"\n[STEP 4] Tuning {best_name} with GridSearchCV...")

param_grids = {
    'RandomForest': {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 20]},
    'SVM_RBF': {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']},
    'LogisticRegression': {'clf__clf__C': [0.1, 1, 10], 'clf__clf__solver': ['lbfgs', 'saga']}
}

grid = GridSearchCV(
    candidates[best_name],
    param_grids.get(best_name, {}),
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X, y)
best_model = grid.best_estimator_
best_params = grid.best_params_
print(f"[INFO] Best params: {best_params}")

# Step 8: Final evaluation on full train
y_pred = best_model.predict(X)
final_acc = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted', zero_division=0)
recall = recall_score(y, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

print(f"\n[STEP 5] Final Evaluation Metrics")
print(f"  Accuracy:  {final_acc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"\n[Classification Report]\n{classification_report(y, y_pred)}")

# Step 9: Feature importance
if 'RandomForest' in best_name:
    try:
        importances = best_model.named_steps['clf'].feature_importances_
        top_k = np.argsort(importances)[::-1][:10]
        print(f"\n[Feature Importance] Top 10 embedding dims: {top_k.tolist()}")
    except Exception:
        pass

# Step 10: Save versioned model
acc_pct = int(final_acc * 100)
model_filename = f"resume_classifier_acc{acc_pct}.pkl"
model_path = os.path.join(OUTPUT_DIR, model_filename)
# Also save as canonical name for inference
canonical_path = os.path.join(OUTPUT_DIR, 'resume_classifier.pkl')

joblib.dump(best_model, model_path)
joblib.dump(best_model, canonical_path)
print(f"\n[OK] Model saved: {model_path}")
print(f"[OK] Canonical model saved: {canonical_path}")

# Step 11: Save metadata/report
metadata = {
    'module': 'Resume Intelligence',
    'model_name': best_name,
    'cv_scores': cv_results[best_name].tolist(),
    'mean_accuracy': float(cv_results[best_name].mean()),
    'std_accuracy': float(cv_results[best_name].std()),
    'final_accuracy': float(final_acc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'best_params': best_params,
    'model_file': model_filename,
    'timestamp': datetime.now().isoformat(),
    'embedding_model': 'all-MiniLM-L6-v2',
    'dataset_rows': int(len(df))
}
with open(METRICS_PATH, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"[OK] Report saved: {METRICS_PATH}")

# Update global metrics
GLOBAL_METRICS_PATH = '../models/training_metrics.json'
global_metrics = {}
if os.path.exists(GLOBAL_METRICS_PATH):
    with open(GLOBAL_METRICS_PATH, 'r') as f:
        global_metrics = json.load(f)
global_metrics['Resume Intelligence'] = metadata
with open(GLOBAL_METRICS_PATH, 'w') as f:
    json.dump(global_metrics, f, indent=4)
print(f"[OK] Global metrics updated at {GLOBAL_METRICS_PATH}")

print("\n[OK] MODULE 1 TRAINING COMPLETE")
