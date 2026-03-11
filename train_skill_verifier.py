import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

print("--- TRAINING SKILL VERIFICATION ENGINE (MODULE 3) ---")

# 1. Dataset Generation: AST Code Metrics vs Skill Credibility Performance
# Target: Skill Credibility Score (0-100)
np.random.seed(42)
n_samples = 1500

# Artificial features simulating code parser insights
complexity = np.random.poisson(3, n_samples)
loops = np.random.poisson(2, n_samples)
loc = np.random.normal(50, 15, n_samples)  # lines of code
var_count = np.random.poisson(5, n_samples)

# Derived skill credibility logic with noise added to simulate real datasets
# High complexity + long code = generally worse
# Moderate complexity + elegant looping = better
credibility_scores = []
for i in range(n_samples):
    base_score = 75
    
    comp = complexity[i]
    loop_cnt = loops[i]
    lines = loc[i]
    
    # Heuristics that the ML will learn:
    if comp > 5:
        base_score -= (comp - 5) * 4
    if loop_cnt > 3:
        base_score -= (loop_cnt - 3) * 3
    if lines > 100:
        base_score -= (lines - 100) * 0.2
        
    noise = np.random.normal(0, 5)
    score = np.clip(base_score + noise, 0, 100)
    credibility_scores.append(score)

X = pd.DataFrame({
    'complexity': complexity,
    'loops': loops,
    'loc': loc,
    'var_count': var_count
})
y = np.array(credibility_scores)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training (Regression)
print("Training Random Forest Regressor for skill credibility prediction...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 4. Evaluation Metrics
y_pred = rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- EVALUATION METRICS ---")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Confidence interval approximation (Standard Deviation of predictions)
conf_interval = np.std(y_test - y_pred) * 1.96  # 95% CI margin
print(f"95% Confidence Interval Span: ±{conf_interval:.2f} points")

# 5. Save the trained model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, os.path.join("models", "skill_verifier_rf.pkl"))
joblib.dump(scaler, os.path.join("models", "skill_scaler.pkl"))

print("\nModels saved to models/skill_verifier_rf.pkl and models/skill_scaler.pkl")
