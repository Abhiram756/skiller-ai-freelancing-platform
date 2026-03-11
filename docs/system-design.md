# Skiller ? System Design

## Architecture Overview

```
+-------------------------------------------------+
|                  FRONTEND                        |
|         demo.html (HTML + JS + CSS)              |
|         Socket.IO Client                         |
+---------------------+---------------------------+
                      | WebSocket / HTTP
+---------------------?---------------------------+
|                  BACKEND                         |
|         Node.js + Express (port 3000)            |
|         Socket.IO Server                         |
|         JWT Auth + bcrypt                        |
+--------+------------------------+---------------+
         | Mongoose ODM            | HTTP (fetch)
+--------?--------+    +----------?--------------+
|  MongoDB Atlas  |    |  Python FastAPI          |
|  (Database)     |    |  ML Microservices        |
|                 |    |  (port 8000)             |
|  Collections:   |    |                          |
|  - users        |    |  Endpoints:              |
|  - jobs         |    |  /analyze-resume         |
|                 |    |  /match-talent           |
+-----------------+    |  /verify-skill           |
                       |  /detect-fraud           |
                       |  /trust-score            |
                       |  /health                 |
                       +----------+---------------+
                                  | joblib.load()
                       +----------?---------------+
                       |   Trained ML Models       |
                       |                          |
                       |  resume_classifier.pkl   |
                       |  skill_verifier.pkl      |
                       |  fraud_detector_rf.pkl   |
                       |  fraud_detector_iso.pkl  |
                       |  trust_scorer.pkl        |
                       +--------------------------+
```

## ML Module Details

| Module | Algorithm | Input | Output |
|---|---|---|---|
| Resume Intelligence | RandomForest + SentenceTransformer | PDF/Text | Role + ATS Score |
| Talent Matching | Cosine Similarity (embeddings) | Job desc + profiles | Top-K matches |
| Skill Verification | GradientBoosting (AST features) | Code string | Score 0-100 |
| Fraud Detection | RandomForest + IsolationForest | Behaviour features | Risk score |
| Trust Score | GradientBoosting | Activity metrics | Trust score 0-100 |

## Security Design

- Passwords: bcrypt (10 salt rounds)
- Sessions: JWT tokens (7-day expiry)
- Secrets: Environment variables only
- DB: MongoDB Atlas with IP whitelisting

## Real-Time Features

Socket.IO events:
- `signup` / `login` -> `authSuccess` / `authError`
- `analyzeResumeNLP` -> `resumeNLPResult`
- `runSkillVerification` -> `skillVerificationResult`
- `runFraudCheck` -> `fraudCheckResult`
- `analyzeTrustScore` -> `trustScoreResult`

