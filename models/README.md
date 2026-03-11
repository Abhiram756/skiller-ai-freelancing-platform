# Models

This folder stores trained ML model files (`.pkl`).

**Models are NOT included in the public repository** due to file size constraints.

## How to generate models locally

```bash
cd training_scripts

# Generate datasets first
python 0_prepare_resume_data.py

# Train each module
python 1_train_resume_intelligence.py
python 2_train_skill_verifier.py
python 3_train_fraud_detector.py
python 4_train_trust_scorer.py
```

Trained `.pkl` files will be saved automatically to:
- `models/resume/`
- `models/skill/`
- `models/fraud/`
- `models/trust/`
