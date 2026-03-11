import sys
import json
import re
from collections import Counter
import base64
import io
import os

try:
    import joblib
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

# Simulated ML/NLP Pipeline for Resume Analysis

# Define Industry Mapping Knowledge Graph (Heuristic for this demo)
KNOWLEDGE_GRAPH = {
    "frontend": ["react", "vue", "angular", "html", "css", "javascript", "typescript", "tailwind"],
    "backend": ["node.js", "python", "java", "django", "express", "spring", "go", "sql", "mongodb"],
    "design": ["ui", "ux", "figma", "adobe xd", "wireframing", "prototyping", "photoshop"],
    "devops": ["aws", "docker", "kubernetes", "ci/cd", "jenkins", "linux", "bash"],
    "ai_ml": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "pandas"]
}

TECH_KEYWORDS = set(item for sublist in KNOWLEDGE_GRAPH.values() for item in sublist)

def extract_text_from_file(file_data, file_type, filename):
    try:
        # Check by extension or file type
        is_pdf = 'pdf' in file_type.lower() or filename.lower().endswith('.pdf')
        is_docx = 'wordprocessingml' in file_type.lower() or 'msword' in file_type.lower() or filename.lower().endswith(('.doc', '.docx'))
        
        if is_pdf and PdfReader:
            pdf = PdfReader(io.BytesIO(file_data))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
        elif is_docx and docx:
            doc = docx.Document(io.BytesIO(file_data))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        else:
            # Fallback for plain text or txt
            return file_data.decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_skills_nlp(text):
    text_lower = text.lower()
    # Tokenize simply for demo purposes
    words = re.findall(r'\b\S+\b', text_lower)
    
    found_skills = set()
    for word in words:
        if word in TECH_KEYWORDS:
            found_skills.add(word.capitalize())
            
    # Check multi-word skills (like "machine learning")
    for keyword in TECH_KEYWORDS:
        if " " in keyword and keyword in text_lower:
            found_skills.add(keyword.title())
            
    return list(found_skills)

def analyze_resume(text):
    extracted_skills = extract_skills_nlp(text)
    
    # 1. Calculate Concept Density / Resume Strength
    base_score = 40
    skill_impact = min(len(extracted_skills) * 5, 45) # Up to 45 pts from skills
    
    # Simple metric extraction (e.g. years of experience, percentages)
    metrics_present = len(re.findall(r'\b\d+%\b|\byears?\b|\$?\d+[kK]\b', text.lower()))
    metric_impact = min(metrics_present * 3, 15) # Up to 15 pts for using metrics
    
    total_score = base_score + skill_impact + metric_impact
    total_score = min(total_score, 100)
    
    # 2. Determine Primary Domain (Clustering / ML Prediction)
    primary_domain = "General"
    model_confidence = 0.0
    
    model_path = os.path.join("models", "resume_classifier.pkl")
    vectorizer_path = os.path.join("models", "resume_vectorizer.pkl")
    
    if ML_AVAILABLE and os.path.exists(model_path) and os.path.exists(vectorizer_path):
        import joblib
        clf = joblib.load(model_path)
        vec = joblib.load(vectorizer_path)
        
        # Predict using the trained model
        X_infer = vec.transform([text])
        prediction = clf.predict(X_infer)[0]
        primary_domain = prediction
        
        # Adjust base score based on model confidence if proba is available
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_infer)[0]
            model_confidence = max(probs)
            # Add similarity/confidence metric (0-10) to total score
            total_score += min(model_confidence * 10, 10)
    else:
        # Fallback to heuristics
        domain_counts = {domain: 0 for domain in KNOWLEDGE_GRAPH}
        for skill in extracted_skills:
            skill_lower = skill.lower()
            for domain, keywords in KNOWLEDGE_GRAPH.items():
                if skill_lower in keywords:
                    domain_counts[domain] += 1
        primary_domain = max(domain_counts, key=domain_counts.get) if any(domain_counts.values()) else "General"
    
    total_score = min(total_score, 100)
    
    # 3. Identify Skill Gaps
    gaps = []
    if primary_domain != "General" and primary_domain.lower() in [k.lower() for k in KNOWLEDGE_GRAPH]:
        domain_key = primary_domain.lower()
        domain_keywords = KNOWLEDGE_GRAPH.get(domain_key, [])
        extracted_lower = [s.lower() for s in extracted_skills]
        missing = [k for k in domain_keywords if k not in extracted_lower]
        if missing:
            gaps = [m.capitalize() for m in missing[:3]] # Suggest top 3 missing
            
    # Feedback Generation
    feedback = []
    if metric_impact < 5:
        feedback.append("Increase your impact by using numbers/metrics (e.g., 'improved 20%').")
    if len(extracted_skills) < 3:
        feedback.append("Your resume lacks hard technical keywords. Ensure ATS systems can read your skills.")
    if model_confidence < 0.4 and primary_domain == "General":
        feedback.append("Your skills seem scattered. Try tailoring your resume to a specific role.")
    elif model_confidence > 0:
        feedback.append(f"ML Model successfully matched you to {primary_domain} with {int(model_confidence*100)}% confidence.")
        
    return {
        "status": "success",
        "resume_strength_score": int(total_score),
        "extracted_skills": extracted_skills,
        "primary_role": primary_domain.replace("_", " ").title(),
        "skill_gaps": gaps,
        "actionable_feedback": feedback
    }


if __name__ == "__main__":
    try:
        # Increase limit if necessary, but sys.stdin.read() handles arbitrary length
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No resume text provided"}))
            sys.exit(0)
            
        data = json.loads(input_data)
        file_base64 = data.get("file_data", "")
        file_type = data.get("file_type", "")
        filename = data.get("filename", "")
        
        if file_base64:
            file_bytes = base64.b64decode(file_base64)
            resume_text = extract_text_from_file(file_bytes, file_type, filename)
        else:
            resume_text = data.get("resume_text", "")
            
        results = analyze_resume(resume_text)
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps({"status": "error", "error": f"NLP Engine Failure: {str(e)}"}))
