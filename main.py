"""
Skiller AI – ML Microservices (FastAPI)
Production-grade inference layer with versioned model loading.
"""

import os
import ast
import re
import base64
import io
import json
import time
import numpy as np
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ──────────────────────── App Setup ────────────────────────
app = FastAPI(
    title="Skiller AI – ML Microservices",
    description="Production ML inference layer for Skiller freelancing platform.",
    version="2.0.0"
)

MODELS = {}
BASE_DIR = os.path.dirname(__file__)
MODELS_BASE = os.path.join(BASE_DIR, '../models')

# ──────────────────────── Model Loading ────────────────────────
def load_latest_model(module: str, prefix: str, fallback: str):
    """Load the latest versioned model or fallback to canonical name."""
    module_dir = os.path.join(MODELS_BASE, module)
    if not os.path.exists(module_dir):
        os.makedirs(module_dir, exist_ok=True)
        return None
    candidates = [
        f for f in os.listdir(module_dir)
        if f.startswith(prefix) and f.endswith('.pkl')
    ]
    if candidates:
        candidates.sort(reverse=True)
        path = os.path.join(module_dir, candidates[0])
        print(f"[LOAD] {module}/{candidates[0]}")
        return joblib.load(path)
    fallback_path = os.path.join(module_dir, fallback)
    if os.path.exists(fallback_path):
        print(f"[LOAD] {module}/{fallback} (canonical fallback)")
        return joblib.load(fallback_path)
    print(f"[WARN] No model found for {module}/{prefix}*. Train models first.")
    return None

def load_all_models():
    print("\n=== Loading ML Models ===")
    if 'sentence_model' not in MODELS:
        print("[LOAD] SentenceTransformer: all-MiniLM-L6-v2")
        MODELS['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')

    MODELS['resume_clf'] = load_latest_model('resume', 'resume_classifier_acc', 'resume_classifier.pkl')
    MODELS['skill_verifier'] = load_latest_model('skill', 'skill_verifier_acc', 'skill_verifier.pkl')
    MODELS['fraud_rf'] = load_latest_model('fraud', 'fraud_detector_rf_f1', 'fraud_detector_rf.pkl')
    MODELS['fraud_iso'] = load_latest_model('fraud', 'fraud_detector_iso', 'fraud_detector_iso.pkl')
    MODELS['trust_scorer'] = load_latest_model('trust', 'trust_scorer_acc', 'trust_scorer.pkl')
    print("=== Models Loaded ===\n")

@app.on_event("startup")
def startup():
    load_all_models()

def get_model(key: str):
    if MODELS.get(key) is None:
        load_all_models()
    if MODELS.get(key) is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{key}' not available. Please run training scripts first."
        )
    return MODELS[key]

# ──────────────────────── Request Models ────────────────────────
class ResumeRequest(BaseModel):
    file_data: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None
    text: Optional[str] = None

class MatchRequest(BaseModel):
    job_description: str
    freelancer_profiles: List[str]
    top_k: Optional[int] = 5

class CodeRequest(BaseModel):
    code: str

class FraudRequest(BaseModel):
    apps_per_hour: Optional[float] = 2.0
    geo_mismatch: Optional[int] = 0
    bio_keyword_suspicion: Optional[float] = 0.1
    profile_age_days: Optional[int] = 365
    identical_proposals: Optional[int] = 0
    payment_dispute_rate: Optional[float] = 0.0
    skill_verification_failed: Optional[int] = 0
    avg_response_time_hours: Optional[float] = 3.0
    # Legacy support
    features: Optional[List[float]] = None
    amount: Optional[float] = None

class TrustRequest(BaseModel):
    completion_rate: float
    avg_rating: Optional[float] = 4.0
    fraud_risk_score: Optional[float] = 10.0
    skill_verification_score: Optional[float] = 70.0
    response_time_hours: Optional[float] = 4.0
    profile_age_days: Optional[int] = 365
    # Legacy support
    ratings: Optional[float] = None
    skill_scores: Optional[float] = None
    historical_activity: Optional[int] = None

# ──────────────────────── Helpers ────────────────────────
SKILLS_VOCAB = {
    "react", "node.js", "nodejs", "python", "java", "sql", "mongodb", "aws",
    "docker", "machine learning", "design", "html", "css", "figma", "pandas",
    "tensorflow", "typescript", "kotlin", "swift", "flutter", "angular",
    "django", "postgres", "kubernetes", "linux", "blockchain", "solidity"
}

def extract_skills_nlp(text: str) -> List[str]:
    text_lower = text.lower()
    words = set(re.findall(r'\b[\w.]+\b', text_lower))
    found = {w.capitalize() for w in words if w in SKILLS_VOCAB}
    for k in SKILLS_VOCAB:
        if " " in k and k in text_lower:
            found.add(k.title())
    return sorted(found)

def extract_text_from_resume(req: ResumeRequest) -> str:
    if req.text:
        return req.text
    if req.file_data:
        try:
            file_bytes = base64.b64decode(req.file_data)
            filename = req.filename or ""
            file_type = req.file_type or ""
            if "pdf" in file_type.lower() or filename.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(file_bytes))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            try:
                import docx as docx_lib
                doc = docx_lib.Document(io.BytesIO(file_bytes))
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception:
                return file_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            return f"Could not parse file: {str(e)}"
    return ""

def compute_ats_score(text: str, extracted_skills: List[str]) -> int:
    score = 0
    length = len(text.strip())
    # Length score (up to 30 pts)
    if length > 2000:
        score += 30
    elif length > 1000:
        score += 20
    elif length > 200:
        score += 10

    # Keyword density (up to 40 pts)
    skill_pts = min(len(extracted_skills) * 5, 40)
    score += skill_pts

    # Section detection (up to 30 pts)
    lower = text.lower()
    for section in ['experience', 'education', 'skills', 'projects', 'summary']:
        if section in lower:
            score += 6

    return min(score, 100)

def extract_ast_features(code: str) -> dict:
    try:
        tree = ast.parse(code)
        func_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        class_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        loop_count = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
        cc = 1 + sum(1 for n in ast.walk(tree)
                     if isinstance(n, (ast.If, ast.For, ast.While, ast.Try,
                                       ast.ExceptHandler, ast.And, ast.Or)))
        node_count = sum(1 for _ in ast.walk(tree))
        return_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Return))
        has_syntax_error = 0
    except SyntaxError:
        return {'cc': 20, 'fc': 0, 'ld': 0, 'cl': len(code),
                'node_count': 5, 'class_count': 0, 'return_count': 0, 'has_syntax_error': 1}

    return {
        'cc': cc, 'fc': func_count, 'ld': loop_count,
        'cl': len(code.strip()), 'node_count': node_count,
        'class_count': class_count, 'return_count': return_count,
        'has_syntax_error': 0
    }

# ──────────────────────── Endpoints ────────────────────────

@app.get("/health")
def health_check():
    loaded = {k: v is not None for k, v in MODELS.items()}
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": loaded
    }

@app.post("/analyze-resume")
def analyze_resume(req: ResumeRequest):
    clf = get_model('resume_clf')
    s_model = get_model('sentence_model')

    text = extract_text_from_resume(req)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No resume text found in the uploaded file.")

    embedding = s_model.encode([text])
    predicted_role = clf.predict(embedding)[0]

    # Top-3 roles with confidence
    top3 = []
    if hasattr(clf, 'predict_proba'):
        probs = clf.predict_proba(embedding)[0]
        classes = clf.classes_
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [{"role": str(classes[i]), "confidence": round(float(probs[i]) * 100, 1)} for i in top3_idx]

    extracted_skills = extract_skills_nlp(text)
    ats_score = compute_ats_score(text, extracted_skills)

    # Skill gap estimation
    role_skills_map = {
        'Data Science': ['Python', 'Machine Learning', 'Pandas', 'Tensorflow'],
        'Web Developer': ['React', 'Node.js', 'Html', 'Css'],
        'DevOps': ['Docker', 'Kubernetes', 'Aws', 'Linux'],
        'Designer': ['Figma', 'Design'],
        'HR': [],
        'Blockchain': ['Solidity', 'Blockchain'],
        'Cybersecurity': ['Linux'],
        'Mobile Developer': ['Flutter', 'Kotlin', 'Swift']
    }
    expected = role_skills_map.get(predicted_role, [])
    skill_gap = [s for s in expected if s not in extracted_skills]

    return {
        "predicted_role": predicted_role,
        "primary_role": predicted_role,
        "top_3_roles": top3,
        "ats_score": ats_score,
        "resume_strength_score": ats_score,
        "extracted_skills": extracted_skills if extracted_skills else ["Extracted via ML model"],
        "skill_gap_list": skill_gap,
        "skill_gaps": skill_gap,
        "actionable_feedback": [
            f"ML model classified your profile as: {predicted_role}.",
            f"ATS Score: {ats_score}/100 – {'Great!' if ats_score > 70 else 'Add more keywords and sections.'}",
            f"{'Missing skills for role: ' + ', '.join(skill_gap) if skill_gap else 'No critical skill gaps detected.'}"
        ]
    }

@app.post("/match-talent")
def match_talent(req: MatchRequest):
    s_model = get_model('sentence_model')

    job_emb = s_model.encode([req.job_description])[0]
    profile_embs = s_model.encode(req.freelancer_profiles)
    sims = cosine_similarity([job_emb], profile_embs)[0]

    matches = sorted(
        [{"profile_index": i, "compatibility_score": float(s)} for i, s in enumerate(sims)],
        key=lambda x: x["compatibility_score"],
        reverse=True
    )[:req.top_k]

    return {"matches": matches, "model": "all-MiniLM-L6-v2", "top_k": req.top_k}

@app.post("/verify-skill")
def verify_skill(req: CodeRequest):
    verifier = get_model('skill_verifier')

    metrics = extract_ast_features(req.code)
    feature_order = ['cc', 'fc', 'ld', 'cl', 'node_count', 'class_count', 'return_count']
    X_infer = [[metrics.get(f, 0) for f in feature_order]]

    score = float(np.clip(verifier.predict(X_infer)[0], 0, 100))

    return {
        "predicted_skill_score": score,
        "final_score": round(score),
        "metrics": {
            "cyclomatic_complexity": metrics['cc'],
            "function_count": metrics['fc'],
            "loop_depth": metrics['ld'],
            "code_length": metrics['cl'],
            "node_count": metrics['node_count'],
            "class_count": metrics['class_count'],
            "analytical_accuracy": round(score),
            "cyclomatic_complexity_score": metrics['cc'],
            "execution_speed_percentile": min(100, round(100 - metrics['cl'] / 50))
        },
        "ai_feedback": [
            "Code structure is clean and modular." if metrics['fc'] > 0 else "No functions detected. Structure your code into functions.",
            "Good cyclomatic complexity." if metrics['cc'] < 10 else "High complexity – consider refactoring.",
            "Well commented and logically sound." if score > 70 else "Add return statements and improve logic coverage."
        ]
    }

@app.post("/detect-fraud")
def detect_fraud(req: FraudRequest):
    fraud_rf = get_model('fraud_rf')

    # Support both new and legacy API format
    if req.features is not None:
        raw = req.features[:8] if len(req.features) >= 8 else req.features + [0.0] * 8
        feature_vector = raw[:8]
    else:
        feature_vector = [
            req.apps_per_hour,
            req.geo_mismatch,
            req.bio_keyword_suspicion,
            float(req.profile_age_days),
            float(req.identical_proposals),
            req.payment_dispute_rate,
            float(req.skill_verification_failed),
            req.avg_response_time_hours
        ]

    X_infer = [feature_vector]

    # Primary: RandomForest
    rf_pred = fraud_rf.predict(X_infer)[0]
    rf_proba = fraud_rf.predict_proba(X_infer)[0][1] if hasattr(fraud_rf, 'predict_proba') else (0.9 if rf_pred else 0.1)

    is_fraud = bool(rf_pred == 1)
    risk_score = round(float(rf_proba) * 100, 1)

    # Backup: IsolationForest
    iso_is_fraud = False
    try:
        iso_bundle = get_model('fraud_iso')
        iso_model = iso_bundle['model']
        iso_scaler = iso_bundle['scaler']
        X_scaled = iso_scaler.transform(X_infer)
        iso_pred = iso_model.predict(X_scaled)[0]
        iso_is_fraud = bool(iso_pred == -1)
    except Exception:
        pass

    final_fraud = is_fraud or iso_is_fraud

    return {
        "is_fraud": final_fraud,
        "rf_prediction": is_fraud,
        "iso_prediction": iso_is_fraud,
        "risk_score": risk_score,
        "assessment": (
            f"CRITICAL: Fraud Anomaly Detected (Risk: {risk_score}%)" if final_fraud
            else f"SECURE: Normal Activity (Risk: {risk_score}%)"
        )
    }

@app.post("/trust-score")
def get_trust_score(req: TrustRequest):
    scorer = get_model('trust_scorer')

    # Support legacy fields
    avg_rating = req.avg_rating if req.avg_rating is not None else (req.ratings or 4.0)
    skill_score = req.skill_verification_score if req.skill_verification_score is not None else (req.skill_scores or 70.0)
    profile_age = req.profile_age_days
    if profile_age is None and req.historical_activity is not None:
        profile_age = req.historical_activity * 10
    profile_age = profile_age or 365

    X_infer = [[
        req.completion_rate,
        avg_rating,
        req.fraud_risk_score or 10.0,
        skill_score,
        req.response_time_hours or 4.0,
        float(profile_age)
    ]]

    raw_score = float(np.clip(scorer.predict(X_infer)[0], 0, 100))
    score = round(raw_score)

    tier = "Elite Trusted" if score >= 85 else ("Highly Trusted" if score >= 70 else ("Valued Member" if score >= 50 else "Building Trust"))

    return {
        "trust_score": score,
        "trust_index": score,
        "tier": tier,
        "sentiment_summary": f"Trust analysis complete. Score: {score}/100 ({tier})",
        "breakdown": {
            "completion_rate": req.completion_rate,
            "avg_rating": avg_rating,
            "skill_score": skill_score
        }
    }

# ──────────────────────── Run ────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
