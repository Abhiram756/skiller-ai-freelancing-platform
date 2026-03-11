"""
2_train_skill_verifier.py
Code quality evaluation via AST features → Multi-model skill score predictor.
"""

import os
import ast
import json
import random
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = '../models/skill'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MODULE 3: Skill Verification – Training")
print("=" * 60)

# --- Code snippets database for synthetic dataset ---
CODE_TEMPLATES = {
    'high': [
        """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
        """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    return result + left[i:] + right[j:]
""",
        """
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1
    def put(self, key, val):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            del self.cache[self.order.pop(0)]
        self.cache[key] = val
        self.order.append(key)
"""
    ],
    'medium': [
        """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
""",
        """
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]

def count_vowels(s):
    vowels = set('aeiouAEIOU')
    count = 0
    for c in s:
        if c in vowels:
            count += 1
    return count
""",
        """
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
"""
    ],
    'low': [
        """
x = 10
y = 20
print(x + y)
""",
        """
a = [1, 2, 3, 4, 5]
total = 0
for i in a:
    total = total + i
print(total)
""",
        """
name = "Alice"
print("Hello " + name)
"""
    ]
}

def extract_ast_features(code):
    """Extract comprehensive AST-based code metrics."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {'cc': 15, 'fc': 1, 'ld': 3, 'cl': len(code), 'node_count': 10, 'class_count': 0, 'return_count': 1}

    func_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    class_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    loop_nodes = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
    cc = 1 + sum(1 for n in ast.walk(tree)
                 if isinstance(n, (ast.If, ast.For, ast.While, ast.Try,
                                   ast.ExceptHandler, ast.And, ast.Or)))
    node_count = sum(1 for _ in ast.walk(tree))
    return_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Return))

    return {
        'cc': cc,
        'fc': func_count,
        'ld': loop_nodes,
        'cl': len(code.strip()),
        'node_count': node_count,
        'class_count': class_count,
        'return_count': return_count
    }

# Generate synthetic skill dataset
print("\n[STEP 1] Generating synthetic code evaluation dataset...")
rows = []
score_map = {'high': (75, 95), 'medium': (45, 74), 'low': (10, 44)}

for level, templates in CODE_TEMPLATES.items():
    for _ in range(600):
        code = random.choice(templates)
        feats = extract_ast_features(code)
        low, high = score_map[level]
        score = random.randint(low, high)
        rows.append({**feats, 'skill_score': score})

df = pd.DataFrame(rows)
print(f"[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Score distribution:\n{df['skill_score'].describe()}")

feature_cols = ['cc', 'fc', 'ld', 'cl', 'node_count', 'class_count', 'return_count']
X = df[feature_cols].values
y = df['skill_score'].values

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

candidates = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED)
}

print("\n[STEP 2] Cross-validating regressors...")
cv_r2 = {}
for name, model in candidates.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    cv_r2[name] = scores
    print(f"  {name}: R2 mean={scores.mean():.4f} std={scores.std():.4f}")

best_name = max(cv_r2, key=lambda k: cv_r2[k].mean())
print(f"\n[INFO] Best: {best_name}")

# GridSearchCV
print(f"\n[STEP 3] Tuning {best_name}...")
param_grids = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    'GradientBoosting': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
}
grid = GridSearchCV(candidates[best_name], param_grids[best_name], cv=cv, scoring='r2', n_jobs=-1)
grid.fit(X, y)
best_model = grid.best_estimator_
print(f"[INFO] Best params: {grid.best_params_}")

y_pred = best_model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\n[STEP 4] Evaluation:")
print(f"  MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

# Feature importances
importances = best_model.feature_importances_
for fname, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {fname}: {imp:.4f}")

# Save model
r2_pct = int(r2 * 100)
model_filename = f"skill_verifier_acc{r2_pct}.pkl"
model_path = os.path.join(OUTPUT_DIR, model_filename)
canonical_path = os.path.join(OUTPUT_DIR, 'skill_verifier.pkl')
joblib.dump(best_model, model_path)
joblib.dump(best_model, canonical_path)
print(f"\n[OK] Saved: {model_path}")

metadata = {
    'module': 'Skill Verification',
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
gm['Skill Verification'] = metadata
with open(GLOBAL_METRICS, 'w') as f:
    json.dump(gm, f, indent=4)

print("[OK] MODULE 3 TRAINING COMPLETE")
