import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("--- TRAINING REPUTATION INTELLIGENCE ENGINE (MODULE 5) ---")

# 1. Dataset Generation: Activity, Verified Identity, Sentiment -> Trust Index
np.random.seed(42)
n_samples = 2000

# Feature 1: Identity Score (0-30) 
identity_verified = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
identity_score = np.where(identity_verified, 30, 10) # 10 for account created

# Feature 2: Engagement (Number of Jobs completed)
completed_jobs = np.random.poisson(10, n_samples)
engagement_score = np.minimum(completed_jobs * 2, 20)

# Feature 3: Review Sentiment (0 to 50 pts)
avg_review_sentiment = np.random.uniform(1.0, 5.0, n_samples)
sentiment_score = (avg_review_sentiment / 5.0) * 50

# Target Calculation
trust_index = identity_score + engagement_score + sentiment_score
# Add some regression noise based on unrecorded interactions (e.g. deadline metrics)
noise = np.random.normal(0, 3, n_samples)
trust_index = np.clip(trust_index + noise, 0, 100)

X = pd.DataFrame({
    'identity_score': identity_score,
    'engagement_score': engagement_score,
    'sentiment_score': sentiment_score
})
y = trust_index

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
print("Training Gradient Boosting Model for Dynamic Regression...")
gbr_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gbr_model.fit(X_train, y_train)

# 4. Evaluation Metrics
y_pred = gbr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- EVALUATION METRICS ---")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# 5. Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(gbr_model, os.path.join("models", "trust_scorer_gb.pkl"))

print("\nModel saved to models/trust_scorer_gb.pkl")
