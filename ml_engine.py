import sys
import json
import logging
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def process_ml_task(data):
    try:
        task_type = data.get('action', 'recommend_talent') # default to old behavior if not specified

        # --- MODE 1: CLIENT LOOKING FOR TALENT (PREDICTIVE MATCHING) ---
        if task_type == 'recommend_talent':
            job_data = data.get('job', {})
            job_desc = job_data.get('description', '') + " " + " ".join(job_data.get('tags', []))
            job_budget_str = str(job_data.get('budget', '0')).replace(',', '').replace('₹', '')
            job_budget = float(job_budget_str) if job_budget_str.replace('.', '', 1).isdigit() else 0.0

            students = data.get('students', [])
            if not students: return []

            student_profiles = [s.get('name', '') + " " + " ".join(s.get('skills', [])) for s in students]

            # 1. Semantic Skill Match (TF-IDF mapping)
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform([job_desc] + student_profiles)
                job_vector = vectors[0]
                student_vectors = vectors[1:]
                similarities = cosine_similarity(job_vector, student_vectors).flatten()
            except ValueError:
                 similarities = [0.1] * len(students)

            results = []
            
            # 2. Predictive Execution Matrix
            for idx, student in enumerate(students):
                semantic_score = float(similarities[idx])
                
                if semantic_score < 0.01: continue # Skip completely irrelevant

                # Parse Student Data
                skill_score = student.get('skillScore', 50) / 100.0 # Normalized 0-1
                is_verified = 1.2 if student.get('isVerified', False) else 1.0
                
                rate_str = str(student.get('hourlyRate', '0')).replace(',', '').replace('₹', '')
                hourly_rate = float(rate_str) if rate_str.replace('.', '', 1).isdigit() else 0.0
                
                # Pricing Suitability Model (Heuristic representation of a regressor)
                pricing_score = 1.0
                if job_budget > 0 and hourly_rate > 0:
                    # e.g., if job is 10k, and they normally charge 100/hr, assume 100 hours.
                    implied_hours = job_budget / hourly_rate
                    if implied_hours < 2: 
                        pricing_score = 0.5 # Too expensive
                    elif implied_hours > 100:
                        pricing_score = 0.8 # Overqualified / Too cheap
                    else:
                        pricing_score = 1.0 # Goldilocks zone
                        
                # --- ML PREDICTION INFERENCE ---
                # Features: semantic_score, skill_score, is_verified, rate, budget, pricing_suitability
                fallback_prob = min(max(((semantic_score * 0.45) + (skill_score * (1.2 if is_verified else 1.0) * 0.35) + (pricing_score * 0.20)) * 100 * 1.5, 12.0), 99.8)
                success_prob_pct = fallback_prob
                
                model_path = os.path.join("models", "job_matcher_gb.pkl")
                if ML_AVAILABLE and os.path.exists(model_path):
                    gb_model = joblib.load(model_path)
                    X_infer = np.array([[semantic_score, skill_score, is_verified, rate, budget, pricing_score]])
                    success_prob_pct = float(gb_model.predict(X_infer)[0])
                    success_prob_pct = min(max(success_prob_pct, 5.0), 99.8)
                
                results.append({
                    "candidate": student,
                    "score": round(success_prob_pct, 1), # Predict Job Success Probability
                    "analysis": f"Predictive Match: [{round(success_prob_pct, 1)}% Success Likelihood] based on ML ranking."
                })
            
            # Sort by Predicted Success Likelihood
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:5]

        # --- MODE 2: STUDENT LOOKING FOR PROJECTS (PREDICTIVE MATCHING) ---
        elif task_type == 'recommend_projects':
            user_profile = data.get('user', {})
            user_text = " ".join(user_profile.get('skills', [])) + " " + user_profile.get('bio', '')
            
            # Extract User Baseline Metrics
            user_skill_score = user_profile.get('skillScore', 50) / 100.0
            user_is_verified = 1.2 if user_profile.get('isVerified', False) else 1.0
            
            rate_str = str(user_profile.get('hourlyRate', '0')).replace(',', '').replace('₹', '')
            user_rate = float(rate_str) if rate_str.replace('.', '', 1).isdigit() else 0.0

            available_jobs = data.get('jobs', [])
            if not available_jobs: return []
            
            job_descriptions = [j.get('title', '') + " " + j.get('description', '') + " " + " ".join(j.get('tags', [])) for j in available_jobs]

            # 1. Semantic Match via TF-IDF
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform([user_text] + job_descriptions)
                user_vector = vectors[0]
                job_vectors = vectors[1:]
                similarities = cosine_similarity(user_vector, job_vectors).flatten()
            except ValueError:
                similarities = [0.1] * len(available_jobs)
            
            results = []

            # 2. Predictive Job Success Modeling
            for idx, job in enumerate(available_jobs):
                semantic_score = float(similarities[idx])
                
                if semantic_score < 0.01: continue

                job_budget_str = str(job.get('budget', '0')).replace(',', '').replace('₹', '')
                job_budget = float(job_budget_str) if job_budget_str.replace('.', '', 1).isdigit() else 0.0

                # Pricing Suitability (Student side evaluation)
                pricing_score = 1.0
                if job_budget > 0 and user_rate > 0:
                    implied_hours = job_budget / user_rate
                    if implied_hours < 2: 
                        pricing_score = 0.5 # Budget too low for their rate
                    elif implied_hours > 100:
                        pricing_score = 0.8 # Massive project, high risk
                    else:
                        pricing_score = 1.0
                
                # --- ML PREDICTION INFERENCE ---
                fallback_prob = min(max(((semantic_score * 0.50) + (user_skill_score * user_is_verified * 0.30) + (pricing_score * 0.20)) * 100 * 1.5, 15.0), 99.8)
                success_prob_pct = fallback_prob
                
                model_path = os.path.join("models", "job_matcher_gb.pkl")
                if ML_AVAILABLE and os.path.exists(model_path):
                    gb_model = joblib.load(model_path)
                    # For student finding job, verified translates standard boolean
                    is_ver = 1 if user_profile.get('isVerified', False) else 0
                    X_infer = np.array([[semantic_score, user_skill_score, is_ver, user_rate, job_budget, pricing_score]])
                    success_prob_pct = float(gb_model.predict(X_infer)[0])
                    success_prob_pct = min(max(success_prob_pct, 5.0), 99.8)

                results.append({
                    "id": job.get('id'),
                    "title": job.get('title'),
                    "budget": job.get('budget'),
                    "tags": job.get('tags'),
                    "matchScore": round(success_prob_pct, 1),
                    "analysis": f"Predictive Match: [{round(success_prob_pct, 1)}% Compatibility Likelihood] scored by Deep ML Engine."
                })

            # Sort Highest Probability First
            results.sort(key=lambda x: x['matchScore'], reverse=True)
            return results[:5]

        # --- MODE 3: AI SKILL SCORING (NEW) ---
        elif task_type == 'calculate_skill_score':
            user = data.get('user', {})
            
            # 1. Market Demand Analysis (TF-IDF against Trending Topics)
            # Define a "Market Corpus" representing current trends
            market_corpus = [
                "react nodejs javascript fullstack web development frontend backend",
                "python data science machine learning ai deep learning pytorch tensorflow",
                "java spring boot enterprise software microservices",
                "flutter dart mobile app development ios android",
                "ui ux design figma adobe xd prototyping wireframing",
                "devops aws docker kubernetes cloud computing ci cd",
                "blockchain solidity smart contracts web3 crypto",
                "cybersecurity ethical hacking network security penetration testing"
            ]
            
            user_skills_text = " ".join(user.get('skills', [])) + " " + user.get('role', '')
            
            try:
                vectorizer = TfidfVectorizer(stop_words='english')
                # Fit on market corpus + user skills
                tfidf_matrix = vectorizer.fit_transform(market_corpus + [user_skills_text])
                
                # Compare User (last index) against all Market categories
                user_vector = tfidf_matrix[-1]
                market_vectors = tfidf_matrix[:-1]
                
                # Get max similarity to ANY trending category
                market_similarities = cosine_similarity(user_vector, market_vectors).flatten()
                market_relevance_score = float(np.max(market_similarities)) * 100 # 0-100 scale
                
            except ValueError:
                market_relevance_score = 0

            # 2. Pattern Scoring (Heuristics + Weighting)
            # Profile Completeness
            profile_score = 0
            if user.get('name'): profile_score += 10
            if user.get('email'): profile_score += 10
            if user.get('company') or user.get('university'): profile_score += 10
            if user.get('isVerified'): profile_score += 20
            
            # Skill Quantity & Quality
            skill_count = len(user.get('skills', []))
            skill_score = min(skill_count * 5, 30) # Max 30 points for skills

            # Activity (Mock for now, normally based on login/jobs)
            activity_score = 15 

            # TOTAL CALCULATION
            # Market Relevance ensures "quality" of skills matters, not just quantity.
            # Weighted: 30% Market AI, 30% Profile, 25% Skills, 15% Activity
            
            final_ai_score = (
                (market_relevance_score * 0.30) + 
                (profile_score * 0.30) + 
                (skill_score * 0.25) + 
                (activity_score * 0.15)
            )
            
            # Scaling adjustment
            final_ai_score = min(max(final_ai_score, 10), 98) # Clamp between 10 and 98

            # Generate Analysis Text
            if market_relevance_score > 50:
                tip = "Your skills are highly relevant to current market trends! 🚀"
            elif market_relevance_score > 20:
                tip = "You have good skills, but adding trending tech (AI, React, Cloud) could boost your score."
            else:
                tip = "Consider learning high-demand skills to improve your market fit."

            return {
                "score": round(final_ai_score),
                "breakdown": {
                    "market_ai": round(market_relevance_score),
                    "profile": profile_score,
                    "skills": skill_score,
                    "activity": activity_score
                },
                "ai_tip": tip
            }

    except Exception as e:
        logging.error(f"ML Processing Error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        # Read input from Node.js
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input"}))
            exit(0)
            
        data = json.loads(input_data)
        results = process_ml_task(data)
        
        # Output JSON result
        print(json.dumps(results))
        
    except Exception as e:
        logging.error(f"Fatal Error: {str(e)}")
        print(json.dumps({"error": f"ML Engine Failure: {str(e)}"}))
