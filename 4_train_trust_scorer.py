"""
4_train_trust_scorer.py
Freelancer trust/credibility prediction – GradientBoosting + RandomForest.
"""

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = '../models/trust'
DATASET_PATH = '../datasets/trust_dataset.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODULE 5: Trust Score Model – Training")
print("=" * 60)

# Generate trust dataset
print("\n[STEP 1] Generating trust dataset (3000 rows)...")

rows = []
for _ in range(3000):
    completion_rate = random.uniform(0.0, 1.0)
    avg_rating = random.uniform(1.0, 5.0)
    fraud_risk_score = random.uniform(0, 100)
    skill_verification_score = random.uniform(0, 100)
    response_time_hours = random.uniform(0.5, 96)
    profile_age_days = random.randint(0, 2000)

    # Trust score formula (ground truth)
    trust = (
        completion_rate * 30 +
        (avg_rating / 5.0) * 25 +
        (1 - fraud_risk_score / 100) * 20 +
        (skill_verification_score / 100) * 15 +
        min(1.0, 1 / (response_time_hours / 10 + 0.1)) * 5 +
        min(1.0, profile_age_days / 500) * 5
    )
    trust = min(max(trust + np.random.normal(0, 3), 0), 100)

    rows.append({
        'completion_rate': completion_rate,
        'avg_rating': avg_rating,
        'fraud_risk_score': fraud_risk_score,
        'skill_verification_score': skill_verification_score,
        'response_time_hours': response_time_hours,
        'profile_age_days': profile_age_days,
        'trust_score': trust
    })

df = pd.DataFrame(rows)
df.to_csv(DATASET_PATH, index=False)
print(f"[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Trust score stats:\n{df['trust_score'].describe()}")

feature_cols = [
    'completion_rate', 'avg_rating', 'fraud_risk_score',
    'skill_verification_score', 'response_time_hours', 'profile_age_days'
]
X = df[feature_cols].values
y = df['trust_score'].values

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

candidates = {
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
}

print("\n[STEP 2] Cross-validating...")
cv_r2 = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    cv_r2[name] = scores
    print(f"  {name}: R2 mean={scores.mean():.4f} std={scores.std():.4f}")

best_name = max(cv_r2, key=lambda k: cv_r2[k].mean())
print(f"\n[INFO] Best: {best_name}")

print(f"\n[STEP 3] Tuning {best_name}...")
param_grids = {
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10]
    }
}
grid = GridSearchCV(candidates[best_name], param_grids[best_name], cv=cv, scoring='r2', n_jobs=-1)
grid.fit(X, y)
best_model = grid.best_estimator_
print(f"  Best params: {grid.best_params_}")

y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n[STEP 4] Final Metrics:")
print(f"  MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

print("\n[Feature Importance]")
importances = best_model.feature_importances_
for name_feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {name_feat}: {imp:.4f}")

r2_pct = int(r2 * 100)
model_filename = f"trust_scorer_acc{r2_pct}.pkl"
model_path = os.path.join(OUTPUT_DIR, model_filename)
canonical_path = os.path.join(OUTPUT_DIR, 'trust_scorer.pkl')
joblib.dump(best_model, model_path)
joblib.dump(best_model, canonical_path)
print(f"\n[OK] Saved: {canonical_path}")

metadata = {
    'module': 'Trust Score',
    'model_name': best_name,
    'cv_r2_scores': cv_r2[best_name].tolist(),
    'mean_r2': float(cv_r2[best_name].mean()),
    'std_r2': float(cv_r2[best_name].std()),
    'final_r2': float(r2),
    'mse': float(mse),
    'mae': float(mae),
    'best_params': grid.best_params_,
    'feature_cols': feature_cols,
    'model_file': model_filename,
    'timestamp': datetime.now().isoformat()
}
with open(os.path.join(OUTPUT_DIR, 'model_report.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

GLOBAL_METRICS = '../models/training_metrics.json'
gm = json.load(open(GLOBAL_METRICS)) if os.path.exists(GLOBAL_METRICS) else {}
gm['Trust Score'] = metadata
with open(GLOBAL_METRICS, 'w') as f:
    json.dump(gm, f, indent=4)

print("[OK] MODULE 5 TRAINING COMPLETE")
