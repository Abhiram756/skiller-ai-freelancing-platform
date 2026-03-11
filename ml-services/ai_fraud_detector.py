import sys
import json
import time
import os

try:
    import joblib
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def detect_anomalies(user_data):
    anomalies = []
    risk_score = 0
    
    # 1. Identity & Location Anomaly
    # E.g., Claims to be in NYC, but IP is from a known VPN datacenter
    ip_location = user_data.get("current_ip_location", "Unknown")
    profile_location = user_data.get("profile_location", "Unknown")
    
    if ip_location != "Unknown" and profile_location != "Unknown":
        if ip_location.lower() != profile_location.lower():
            anomalies.append(f"Geo-Mismatch: Profile says {profile_location}, IP shows {ip_location}")
            risk_score += 35
            
    # 2. Behavioral Velocity (GNN / Action frequency)
    # E.g., Applying to 50 jobs in 2 minutes
    applications_per_hour = user_data.get("applications_last_hour", 0)
    if applications_per_hour > 20:
        anomalies.append(f"High Velocity Activity: {applications_per_hour} applications in 1 hour (Bot-like)")
        risk_score += 40
        
    # 3. Content Plagiarism / Duplicate Graph Checking
    bio = user_data.get("bio", "").lower()
    suspicious_keywords = ["do my homework", "buy account", "whatsapp me"]
    bio_suspicion_count = 0
    for keyword in suspicious_keywords:
        if keyword in bio:
            anomalies.append(f"Suspicious Phrase Detected: '{keyword}'")
            bio_suspicion_count += 0.5
            risk_score += 25
            
    bio_suspicion = min(bio_suspicion_count, 1.0)
    
    # --- ML ANOMALY INFERENCE ---
    model_path = os.path.join("models", "fraud_detector_iso.pkl")
    if ML_AVAILABLE and os.path.exists(model_path):
        iso_model = joblib.load(model_path)
        geo_mismatch_flag = 1 if (ip_location != "Unknown" and profile_location != "Unknown" and ip_location.lower() != profile_location.lower()) else 0
        
        X_infer = np.array([[geo_mismatch_flag, applications_per_hour, bio_suspicion]])
        # Prediction: 1 normal, -1 anomaly
        prediction = iso_model.predict(X_infer)[0]
        # decision_function: positive is normal, negative is anomaly.
        decision = iso_model.decision_function(X_infer)[0]
        
        # Scale risk_score using decision boundaries
        if prediction == -1:
            anomalies.append("ML Isolation Forest: Highly anomalous behavioral cluster detected.")
            # Convert decision (like -0.1 to -0.3) into a high risk score (60-100)
            ml_risk = min(max(50 + (abs(decision) * 200), 60), 100)
            risk_score = max(risk_score, ml_risk)
        else:
            # Low risk, normal behavior bounds.
            ml_risk = min(max(abs(decision) * 10, 2), 25)
            risk_score = max(risk_score * 0.5, ml_risk) # temper down risk if ML thinks they are okay
            
    # Normalize risk score
    risk_score = min(math_max(risk_score, 0), 100)
    
    # Thresholding
    if risk_score > 70:
        status = "CRITICAL: High Fraud Probability"
        action = "Account Suspended / Flagged for Manual Review"
    elif risk_score > 30:
        status = "WARNING: Suspicious Activity Detected"
        action = "Require Additional Face Authentication"
    else:
        status = "SECURE: Normal Behavior Pattern"
        action = "None"
        risk_score = 2 # Add tiny baseline noise

    return {
        "status": "success",
        "risk_score": risk_score,
        "assessment": status,
        "recommended_action": action,
        "anomalies_detected": anomalies if anomalies else ["No severe anomalies detected in behavior graph."]
    }

def math_max(a, b):
    return a if a > b else b

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            data = json.loads(line)
            req_id = data.get("req_id", None)
            results = detect_anomalies(data)
            if req_id is not None:
                results["req_id"] = req_id
            print(json.dumps(results))
            sys.stdout.flush()
        except Exception as e:
            error_res = {"status": "error", "error": f"Anomaly Detector Crash: {str(e)}"}
            if req_id is not None:
                error_res["req_id"] = req_id
            print(json.dumps(error_res))
            sys.stdout.flush()
