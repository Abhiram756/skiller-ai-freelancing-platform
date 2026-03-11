import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

print("--- TRAINING JOB MATCHING ENGINE (MODULE 2) ---")

# 1. Dataset Generation: Freelancer Attributes vs Job Success
# Features: Semantic_Score, Skill_Score(0-1), Verified(0/1), Hourly_Rate, Budget
# Target: Success_Probability (0-100)
np.random.seed(42)
n_samples = 2000

semantic_scores = np.random.uniform(0.1, 1.0, n_samples)
skill_scores = np.random.uniform(0.4, 0.99, n_samples)
is_verified = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
hourly_rates = np.random.uniform(10, 150, n_samples)
budgets = np.random.uniform(500, 5000, n_samples)

# Derived feature: budget-to-rate ratio (approx hours)
implied_hours = budgets / hourly_rates
pricing_suitability = np.where((implied_hours >= 5) & (implied_hours <= 100), 1.0, 
                      np.where(implied_hours < 5, 0.5, 0.8))

import math
success_labels = []
for i in range(n_samples):
    # Base heuristic: Semantic + Skill + Price Fit + Verified boost
    base = (semantic_scores[i] * 40) + (skill_scores[i] * 30 * (1.2 if is_verified[i] else 1.0)) + (pricing_suitability[i] * 20)
    # Add random noise for ML to learn patterns, not just a hardcoded rule
    noise = np.random.normal(0, 5)
    
    score = base + noise
    success_labels.append(min(max(score, 5.0), 99.0)) # Clamp 5-99

X = np.column_stack((semantic_scores, skill_scores, is_verified, hourly_rates, budgets, pricing_suitability))
y = np.array(success_labels)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Collaborative Filtering / Gradient Boosting Algorithm
print("Training Gradient Boosting Recommendation Model...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
gb_model.fit(X_train, y_train)

# 4. Compute Metrics
y_pred = gb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- EVALUATION METRICS ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# 5. Save the trained model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "job_matcher_gb.pkl")
joblib.dump(gb_model, model_path)

print(f"\nModel saved to {model_path}")
