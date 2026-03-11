import os
import pandas as pd
import numpy as np

os.makedirs('datasets', exist_ok=True)

# 1. Resume Dataset (Mimicking Kaggle Resume Dataset)
def generate_resume_dataset():
    categories = ['Data Science', 'Web Designing', 'HR', 'Advocate', 'Arts', 'Web Developer', 'Database', 'Hadoop', 'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']
    data = []
    for _ in range(500):
        category = np.random.choice(categories)
        resume_text = f"Experience in {category}. Skilled in various tools related to {category}. "
        if category == 'Data Science':
            resume_text += "Python, Machine Learning, Deep Learning, Pandas, Scikit-learn."
        elif category == 'Web Developer':
            resume_text += "HTML, CSS, JavaScript, React, Node.js, Frontend, Backend."
        elif category == 'Blockchain':
            resume_text += "Solidity, Ethereum, Smart Contracts, Web3, Crypto."
        else:
            resume_text += "General professional experience, teamwork, leadership."
            
        data.append({'Resume': resume_text, 'Category': category})
    df = pd.DataFrame(data)
    df.to_csv('datasets/resume_dataset.csv', index=False)
    print("Generated datasets/resume_dataset.csv")

# 2. Fraud Dataset (Mimicking Credit Card Fraud Dataset for Platform Activity)
def generate_fraud_dataset():
    # V1 to V28 features and 'Amount', 'Class'
    n_samples = 2000
    df = pd.DataFrame(np.random.randn(n_samples, 28), columns=[f'V{i}' for i in range(1, 29)])
    df['Amount'] = np.random.exponential(scale=100, size=n_samples)
    # Target: 0 for normal, 1 for fraud (about 5% fraud)
    df['Class'] = np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)
    
    # Induce anomalies
    fraud_indices = df[df['Class'] == 1].index
    df.loc[fraud_indices, 'V1'] += 5.0
    df.loc[fraud_indices, 'V2'] -= 5.0
    
    df.to_csv('datasets/fraud_dataset.csv', index=False)
    print("Generated datasets/fraud_dataset.csv")

# 3. Trust Index Dataset
def generate_trust_dataset():
    n_samples = 1000
    completion_rate = np.random.uniform(0.5, 1.0, n_samples)
    ratings = np.random.uniform(1.0, 5.0, n_samples)
    skill_scores = np.random.uniform(20, 100, n_samples)
    historical_activity = np.random.randint(1, 100, n_samples)
    
    trust_score = (completion_rate * 40) + (ratings * 5) + (skill_scores * 0.2) + (historical_activity * 0.1)
    trust_score = np.clip(trust_score + np.random.normal(0, 5, n_samples), 0, 100)
    
    df = pd.DataFrame({
        'completion_rate': completion_rate,
        'ratings': ratings,
        'skill_scores': skill_scores,
        'historical_activity': historical_activity,
        'trust_score': trust_score
    })
    df.to_csv('datasets/trust_dataset.csv', index=False)
    print("Generated datasets/trust_dataset.csv")

if __name__ == '__main__':
    generate_resume_dataset()
    generate_fraud_dataset()
    generate_trust_dataset()
