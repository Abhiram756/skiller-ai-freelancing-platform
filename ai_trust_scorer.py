import sys
import json
import time
import math
import os

try:
    import joblib
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Simulated Sentiment Keywords (Simple VADER-like heuristic for MVP)
POSITIVE_WORDS = {"excellent", "great", "fast", "good", "amazing", "perfect", "professional", "reliable"}
NEGATIVE_WORDS = {"bad", "slow", "poor", "late", "unprofessional", "terrible", "scam"}

def analyze_sentiment(review_text):
    text = review_text.lower()
    pos_count = sum(1 for word in POSITIVE_WORDS if word in text)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text)
    
    # +1 for positive, -2 for negative (stricter penalty for bad reviews)
    score = pos_count - (neg_count * 2)
    
    # Bound between 1 and 5
    rating = min(max(3 + score, 1), 5)
    return rating

def calculate_time_decay_weight(days_ago):
    # Newton's Law of Cooling / Exponential Decay
    # Older reviews matter less. Half-life ~180 days.
    return math.exp(-0.0038 * days_ago)

def generate_reputation_intelligence(data):
    user = data.get("user", {})
    mock_reviews = data.get("reviews", []) # List of dicts: {"text": "...", "days_ago": 10}
    
    # 1. Base Identity Trust (Max 30)
    identity_score = 10 # Base for account creation
    if user.get("isVerified"):
        identity_score += 20
        
    # 2. Activity & Engagement (Max 20)
    # Using mock metrics for demo if real ones aren't present
    completed_jobs = user.get("jobsCompleted", 5) 
    engagement_score = min(completed_jobs * 2, 20)
    
    # 3. Sentiment & Time-Decay Review Modeling (Max 50)
    review_impact = 0
    if not mock_reviews:
        # Give neutral-positive to new users, but they can't get 100%
        review_impact = 25 
        sentiment_summary = "Neutral (No history)"
    else:
        weighted_sum = 0
        weight_total = 0
        
        for rev in mock_reviews:
            text = rev.get("text", "")
            days_ago = rev.get("days_ago", 30)
            
            # NLP Sentiment
            sentiment_rating = analyze_sentiment(text) 
            # Scale sentiment rating (1-5) to points (0-50) => (rating/5)*50
            points = (sentiment_rating / 5.0) * 50
            
            # Time Decay
            weight = calculate_time_decay_weight(days_ago)
            
            weighted_sum += points * weight
            weight_total += weight
            
        if weight_total > 0:
            review_impact = weighted_sum / weight_total
        else:
            review_impact = 25
            
        sentiment_summary = f"Aggregated {len(mock_reviews)} reviews with time-decay."
        
    # --- ML TRUST INFERENCE ---
    model_path = os.path.join("models", "trust_scorer_gb.pkl")
    if ML_AVAILABLE and os.path.exists(model_path):
        gbr_model = joblib.load(model_path)
        
        # Build DataFrame payload since the model was trained on DataFrame named features
        import pandas as pd
        X_infer = pd.DataFrame([{
            'identity_score': identity_score,
            'engagement_score': engagement_score,
            'sentiment_score': round(review_impact, 2)
        }])
        
        trust_index = float(gbr_model.predict(X_infer)[0])
    else:
        # Final Trust Index Assembly (Fallback Heuristic)
        trust_index = identity_score + engagement_score + review_impact
        
    trust_index = round(min(max(trust_index, 0), 100))
    
    # Determine Tier
    if trust_index >= 90:
        tier = "Elite Trusted"
    elif trust_index >= 70:
        tier = "Verified Reliable"
    elif trust_index >= 50:
        tier = "Standard"
    else:
        tier = "High Risk"

    return {
        "status": "success",
        "trust_index": trust_index,
        "tier": tier,
        "breakdown": {
            "identity_verification": identity_score,
            "engagement_score": engagement_score,
            "sentiment_score": round(review_impact, 1)
        },
        "sentiment_summary": sentiment_summary
    }

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            data = json.loads(line)
            req_id = data.get("req_id", None)
            results = generate_reputation_intelligence(data)
            if req_id is not None:
                results["req_id"] = req_id
            print(json.dumps(results))
            sys.stdout.flush()
        except Exception as e:
            error_res = {"status": "error", "error": f"Trust Engine Error: {str(e)}"}
            if req_id is not None:
                error_res["req_id"] = req_id
            print(json.dumps(error_res))
            sys.stdout.flush()
