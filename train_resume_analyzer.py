import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json
import os

print("--- TRAINING RESUME ANALYZER (MODULE 1) ---")

# 1. Synthetic Dataset Generation Maps to Job Roles
# In a real environment, this would be loaded from a CSV/DB of thousands of resumes.
data = [
    ("Experienced React developer with 3 years building web applications. I know javascript, html, css, and tailwind. I focus on responsive UI.", "Frontend"),
    ("Backend engineer skilled in Node.js, Python, Django, and MongoDB. 5 years of experience building robust APIs and scaling databases.", "Backend"),
    ("Creative UI/UX designer using Figma and Adobe XD. Designing wireframes, user journeys, and prototypes for modern web apps.", "Design"),
    ("DevOps specialist with AWS, Docker, Kubernetes, and CI/CD pipelines. Managing infrastructure as code.", "DevOps"),
    ("Machine learning engineer with experience in TensorFlow, PyTorch, Deep Learning, and NLP models. Data science background.", "AI_ML"),
    ("Vue.js and Tailwind CSS developer creating highly responsive interfaces and single page applications.", "Frontend"),
    ("Java and Spring Boot developer with microservices background and SQL knowledge. Experience with high-throughput backend systems.", "Backend"),
    ("Product designer focused on user research and wireframing in Photoshop. Good eye for aesthetics.", "Design"),
    ("System administrator managing Linux servers, bash scripts for automation, and Jenkins deployments.", "DevOps"),
    ("Data scientist analyzing pandas dataframes and building predictive regression models. Extracting insights from big data.", "AI_ML"),
]

# Artificially expand dataset for train/test split to work properly
X_raw = [item[0] for item in data] * 20
y = [item[1] for item in data] * 20

print(f"Dataset loaded: {len(X_raw)} records.")

# 2. TF-IDF & Embeddings
print("Applying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(X_raw)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Classification Model (Logistic Regression / SVM)
print("Training Classification Model (Logistic Regression)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Optional: Train others like SVM or Random Forest and pick the best
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)

# 5. Add Evaluation Metrics
print("\n--- EVALUATION METRICS ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# 6. Save Trained Models
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "resume_classifier.pkl")
vectorizer_path = os.path.join("models", "resume_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\nModels saved to {model_path} and {vectorizer_path}")
