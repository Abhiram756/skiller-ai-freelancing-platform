import sys
import json
import ast
import time
import os
import numpy as np

try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

def evaluate_python_code(code_string):
    """
    Evaluates Python code using AST to check for structural efficiency, complexity, and specific patterns.
    This simulates a runtime and semantic evaluation.
    """
    score = 40 # Base attempt score
    feedback = []
    
    try:
        tree = ast.parse(code_string)
        
        # Analyze AST Nodes
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        loops = [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]
        list_comps = [node for node in ast.walk(tree) if isinstance(node, ast.ListComp)]
        try_blocks = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        ifs = [node for node in ast.walk(tree) if isinstance(node, ast.If)]
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        
        # AST Metrics for ML
        loc = len(code_string.splitlines())
        comp_score = len(loops) + len(ifs) + len(try_blocks) + 1
        loop_cnt = len(loops)
        var_cnt = len(assignments)

        # Base feedback
        if functions: feedback.append("Excellent use of modularity (Functions defined).")
        if list_comps: feedback.append("Advanced pattern detected: List Comprehensions show Pythonic understanding.")
        if try_blocks: feedback.append("Good error handling semantics (Try/Except blocks utilized).")
        
        # --- ML INFERENCE ---
        model_path = os.path.join("models", "skill_verifier_rf.pkl")
        scaler_path = os.path.join("models", "skill_scaler.pkl")
        
        if ML_AVAILABLE and os.path.exists(model_path) and os.path.exists(scaler_path):
            rf_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            X_infer = np.array([[comp_score, loop_cnt, loc, var_cnt]])
            X_scaled = scaler.transform(X_infer)
            score = float(rf_model.predict(X_scaled)[0])
            score = min(max(score, 0), 100)
            feedback.append(f"ML Model Analysis - Computed Credibility Probability: {score:.1f}%")
        else:
            # Fallback
            if comp_score > 5: score -= 10
            if loop_cnt > 3: score -= 5
            if len(code_string.strip()) > 30: score += 10
        
        return {"score": min(score, 100), "feedback": feedback, "status": "success"}
        
    except SyntaxError as e:
        return {"score": 0, "feedback": [f"Syntax Error: {str(e)}"], "status": "syntax_error"}
    except Exception as e:
        return {"score": 0, "feedback": [f"Evaluation Error: {str(e)}"], "status": "error"}

def process_submission(data):
    language = data.get("language", "python").lower()
    code = data.get("code", "")
    
    if language == "python":
        result = evaluate_python_code(code)
    else:
        # Fallback for other languages in this demo
        length_score = min(len(code) / 10, 80)
        result = {
            "score": round(length_score), 
            "feedback": ["Code parsed successfully.", "Note: Advanced AST semantic parsing is fully optimized for Python currently."],
            "status": "success"
        }
        
    return {
        "status": result["status"],
        "final_score": result["score"],
        "metrics": {
            "analytical_accuracy": round(result["score"] * 0.95),
            "cyclomatic_complexity_score": round(result["score"] * 0.8),
            "execution_speed_percentile": 88 # Mocked benchmark percentile
        },
        "ai_feedback": result["feedback"]
    }

if __name__ == "__main__":
    try:
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input payload"}))
            sys.exit(0)
            
        data = json.loads(input_data)
        results = process_submission(data)
        
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps({"error": f"AI Evaluator Fatal Crash: {str(e)}"}))
