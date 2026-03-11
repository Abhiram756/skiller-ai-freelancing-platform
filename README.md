# 🚀 Skiller AI — Intelligent Freelancing Platform

![Python](https://img.shields.io/badge/Python-Machine%20Learning-blue)
![Node.js](https://img.shields.io/badge/Node.js-Backend-green)
![FastAPI](https://img.shields.io/badge/FastAPI-ML%20Service-teal)
![MongoDB](https://img.shields.io/badge/MongoDB-Database-darkgreen)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Abhiram%20Settybalija-blue?logo=linkedin)](https://www.linkedin.com/in/abhiram-settybalija-48a58035b)
[![Email](https://img.shields.io/badge/Email-abhiram07may%40gmail.com-red?logo=gmail)](mailto:abhiram07may@gmail.com)

Skiller AI is an **AI-driven freelancing intelligence platform** designed to solve real problems faced by freelancers on traditional marketplaces such as Fiverr and Upwork.

Instead of relying purely on bidding systems, Skiller introduces **machine-learning powered matching, resume intelligence, and trust scoring** to connect the right freelancers with the right projects.

The goal is to create a **fair, intelligent, and transparent freelancing ecosystem** where both freelancers and clients benefit.

---

# 📌 Problem Statement

Existing freelancing platforms suffer from several major problems:

• Oversaturated marketplaces where beginners rarely get visibility  
• Poor freelancer–project matching mechanisms  
• High commission rates on freelancer earnings  
• Clients struggling to quickly find the right freelancer  
• Reputation systems that do not accurately reflect real skill

While exploring existing freelancing platforms, it became clear that even **highly skilled freelancers often fail to secure projects** due to inefficient discovery systems.

Skiller AI was created to address these challenges.

---

# 💡 Solution — Skiller AI

Skiller introduces **AI-powered intelligence layers** to improve how freelancers and clients interact on a marketplace platform.

Core innovations include:

• AI-based freelancer and project matching  
• Resume intelligence using NLP models  
• Skill verification using code analysis  
• Fraud detection for suspicious freelancer behavior  
• Trust scoring system for freelancer credibility  
• Intelligent project recommendations

Instead of endless bidding competition, Skiller focuses on **smart matching algorithms that improve project discovery and platform trust.**

---

# 🖼 Platform Preview

### Main Dashboard
![Dashboard](images/dashboard.png)

### Student Dashboard
![Student Dashboard](images/student-dashboard.png)

### Client Dashboard
![Client Dashboard](images/client-dashboard.png)

### AI Talent Matching Engine
![AI Talent Matching](images/ai-talent-matching.png)

### NLP Resume Verification
![Resume Verification](images/resume-verification.png)

### Skill Verification Engine
![Skill Verification](images/skill-verification.png)

### Face Authentication
![Face Authentication](images/face-authentication.png)

### AI Suggested Projects
![AI Projects](images/ai-project-suggestion.png)

---

# 🧠 Machine Learning System

Skiller integrates multiple machine learning modules that analyze freelancer profiles, evaluate credibility, and improve project matching.

Each module operates as a **microservice within a FastAPI-based ML inference layer**.

### ML Pipeline Features

The machine learning system includes:

• Stratified **5-Fold Cross Validation**  
• Multi-model comparison (RandomForest, SVM, Logistic Regression)  
• Hyperparameter tuning using GridSearchCV  
• Versioned model files for reproducibility  
• Automated model loading during inference  
• Training metrics stored for analysis

Each training module produces a **model_report.json** containing evaluation metrics and training parameters.

---

# 🤖 Machine Learning Modules

| Module | Purpose |
|------|------|
| Resume Intelligence | NLP-based resume classification and ATS scoring |
| Talent Matching | Freelancer–project compatibility scoring |
| Skill Verification | Code analysis using AST feature extraction |
| Fraud Detection | Detection of suspicious freelancer behaviour |
| Trust Score Model | Predictive credibility scoring for freelancers |

These modules form the **AI decision layer of the Skiller platform**.

---

# 📊 Dataset Overview

| Module | Dataset Type | Description |
|------|------|------|
| Resume Analyzer | Synthetic resume dataset | Generated using realistic resume fragments |
| Fraud Detector | Behaviour dataset | Contains freelancing fraud indicators |
| Trust Scorer | Freelancer activity dataset | Platform credibility signals |
| Skill Verifier | Code snippet dataset | AST feature extraction |
| Talent Matcher | Text similarity dataset | TF-IDF cosine similarity |

These datasets simulate **real freelancing platform signals while maintaining reproducible training pipelines**.

---

# 🏗 System Architecture

![Architecture Diagram](docs/architecture-diagram.png)

The platform follows a **hybrid full-stack AI architecture** combining traditional web services with machine learning inference layers.

Core system layers include:

• Frontend dashboard interface  
• Node.js / Express backend APIs  
• FastAPI machine learning microservices  
• MongoDB database layer  
• ML model inference engine

This separation allows scalable integration of **AI decision systems within a freelancing marketplace platform**.

---

# ⚙️ Tech Stack

### Backend
• Node.js  
• Express.js  

### Machine Learning
• Python  
• Scikit-learn  
• FastAPI  
• SentenceTransformers  

### Database
• MongoDB  

### Frontend
• HTML  
• CSS  
• JavaScript  

### Tools
• Git & GitHub  
• REST APIs  
• JSON communication  

---

# 📈 Future Roadmap

Planned improvements for Skiller include:

• Deep learning-based skill verification  
• AI proposal generation for freelancers  
• Advanced freelancer recommendation systems  
• Secure payment gateway integration  
• Blockchain-based reputation verification  
• Automated talent discovery systems

---

# 🤝 Collaboration & Contact

Skiller is currently under active development.

If you are interested in **collaboration, research discussions, or startup opportunities**, feel free to connect.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Abhiram%20Settybalija-blue?logo=linkedin)](https://www.linkedin.com/in/abhiram-settybalija-48a58035b)

[![Email](https://img.shields.io/badge/Email-abhiram07may%40gmail.com-red?logo=gmail)](mailto:abhiram07may@gmail.com)

---

# ⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub.
