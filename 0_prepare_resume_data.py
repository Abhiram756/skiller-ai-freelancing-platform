"""
0_prepare_resume_data.py
Resume Dataset Preparation – checks for Kaggle data, else generates synthetic.
"""

import os
import re
import random
import pandas as pd
import numpy as np

DATASET_PATH = '../datasets/Resume.csv'
OUTPUT_PATH = '../datasets/resume_dataset.csv'
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CATEGORIES = [
    'Data Science', 'Web Developer', 'HR', 'Designer',
    'DevOps', 'Blockchain', 'Cybersecurity', 'Mobile Developer'
]

FRAGMENTS = {
    'Data Science': [
        "Developed machine learning models using scikit-learn and TensorFlow",
        "Analyzed large datasets using Pandas and NumPy for business insights",
        "Built NLP pipelines for text classification and sentiment analysis",
        "Deployed ML models using Flask and Docker on AWS EC2",
        "Performed feature engineering and data preprocessing for predictive modeling",
        "Used deep learning CNNs for image recognition tasks",
        "Created data visualization dashboards using Matplotlib and Seaborn",
        "Managed data pipelines using Apache Spark",
        "Trained recommendation systems using collaborative filtering",
        "Applied XGBoost and LightGBM for Kaggle competitions",
        "Worked with SQL databases for data extraction and transformation",
        "Used BERT transformers for NLP classification tasks",
        "Built LSTM models for time series forecasting",
        "Implemented A/B testing frameworks for product teams",
        "Conducted statistical hypothesis testing using SciPy"
    ],
    'Web Developer': [
        "Built responsive web applications using React.js and TypeScript",
        "Developed RESTful APIs using Node.js and Express",
        "Integrated MongoDB as a backend NoSQL database",
        "Implemented authentication using JWT and OAuth2",
        "Created dynamic UI components using Vue.js and Vuex",
        "Optimized frontend performance achieving 98 Lighthouse score",
        "Built full-stack applications using MERN stack",
        "Deployed web applications on Vercel, AWS, and Heroku",
        "Used GraphQL for API data fetching",
        "Developed Progressive Web Apps with service workers",
        "Implemented WebSockets for real-time features",
        "Used Tailwind CSS and Material UI for styling",
        "Worked with CI/CD pipelines using GitHub Actions",
        "Developed microservices using Docker and Kubernetes",
        "Wrote unit and integration tests using Jest and Cypress"
    ],
    'HR': [
        "Managed end-to-end recruitment for technical positions",
        "Conducted structured behavioral interviews and screening",
        "Implemented performance review and appraisal processes",
        "Handled employee onboarding and orientation programs",
        "Developed HR policies and compliance documentation",
        "Managed payroll and benefits administration for 500+ employees",
        "Used SAP HR and Workday HRIS systems",
        "Led talent acquisition initiatives reducing TTH by 30%",
        "Mediated employee conflicts and maintained positive culture",
        "Organized training and professional development workshops",
        "Created job descriptions aligned with competency frameworks",
        "Tracked key HR metrics using Power BI dashboards",
        "Handled visa processing and international mobility",
        "Partnered with business leaders for workforce planning",
        "Managed employer branding and recruitment marketing"
    ],
    'Designer': [
        "Designed user interfaces in Figma and Adobe XD",
        "Created wireframes and high-fidelity prototypes",
        "Conducted user research and usability testing",
        "Developed design systems and component libraries",
        "Worked with developers on handoff and implementation",
        "Designed mobile-first responsive UI layouts",
        "Created brand identities including logos and style guides",
        "Used Photoshop and Illustrator for visual design",
        "Applied UX principles to improve user retention by 40%",
        "Designed marketing materials for digital and print",
        "Created motion graphics using After Effects",
        "Led design sprints and design thinking workshops",
        "Built accessible designs following WCAG guidelines",
        "Collaborated with product managers on feature ideation",
        "Managed multiple client projects in an agency setting"
    ],
    'DevOps': [
        "Managed AWS infrastructure using Terraform and Ansible",
        "Set up CI/CD pipelines using Jenkins and GitHub Actions",
        "Containerized applications using Docker and Kubernetes",
        "Monitored systems using Prometheus, Grafana, and ELK stack",
        "Implemented infrastructure-as-code for cloud deployments",
        "Managed Linux servers and performed performance tuning",
        "Set up automated disaster recovery and backup systems",
        "Implemented security hardening and vulnerability scanning",
        "Managed multi-cloud deployments across AWS and Azure",
        "Optimized CI/CD pipeline reducing build time by 60%",
        "Used Helm for Kubernetes deployment management",
        "Configured load balancers and auto-scaling groups",
        "Implemented secrets management using HashiCorp Vault",
        "Built observability solutions with distributed tracing",
        "Coordinated incident response and postmortem reviews"
    ],
    'Blockchain': [
        "Developed smart contracts using Solidity on Ethereum",
        "Built DeFi protocols including AMMs and lending platforms",
        "Integrated Web3.js and Ethers.js for DApp development",
        "Deployed NFT marketplaces and minting contracts",
        "Audited smart contracts for security vulnerabilities",
        "Worked with IPFS for decentralized file storage",
        "Built cross-chain bridges using Polygon and Avalanche",
        "Used Hardhat and Truffle for contract development and testing",
        "Integrated Chainlink oracles for real-world data feeds",
        "Developed DAO governance mechanisms and tokenomics",
        "Managed blockchain node operations and infrastructure",
        "Created token vesting and staking smart contracts",
        "Worked with Layer-2 solutions including Arbitrum and Optimism",
        "Built NFT royalty distribution systems",
        "Designed cryptographic authentication systems"
    ],
    'Cybersecurity': [
        "Conducted penetration testing on web and mobile applications",
        "Performed network security assessments and vulnerability scans",
        "Used Metasploit, Burp Suite, and Kali Linux for ethical hacking",
        "Implemented SIEM solutions using Splunk and IBM QRadar",
        "Managed enterprise firewall policies and intrusion detection systems",
        "Wrote security policies and compliance documentation for ISO 27001",
        "Responded to security incidents and performed forensic investigations",
        "Developed security awareness training programs for employees",
        "Implemented zero-trust network architecture",
        "Performed red team exercises and adversary simulations",
        "Conducted OWASP Top 10 vulnerability assessments",
        "Managed PKI and certificate lifecycle",
        "Worked with threat intelligence platforms and IOC analysis",
        "Implemented endpoint detection and response solutions",
        "Configured cloud security groups and IAM policies"
    ],
    'Mobile Developer': [
        "Built iOS applications using Swift and SwiftUI",
        "Developed Android applications using Kotlin and Jetpack Compose",
        "Created cross-platform applications using React Native",
        "Used Flutter and Dart for mobile app development",
        "Integrated REST APIs and GraphQL in mobile clients",
        "Implemented push notifications using Firebase Cloud Messaging",
        "Optimized app performance reducing load time by 50%",
        "Published apps to App Store and Google Play Store",
        "Implemented local storage using Core Data and Room DB",
        "Used MVVM architecture pattern for scalable codebases",
        "Integrated payment gateways like Stripe and Razorpay in mobile apps",
        "Built offline-first mobile apps with sync capabilities",
        "Managed app signing, certificates, and build configurations",
        "Wrote unit and UI tests using XCTest and Espresso",
        "Used CI/CD pipelines with Fastlane for mobile deployments"
    ]
}

def inject_noise(text):
    """Inject 5% typos, mixed casing, abbreviations."""
    words = text.split()
    result = []
    for w in words:
        r = random.random()
        if r < 0.02:
            # Swap two adjacent chars
            if len(w) > 3:
                i = random.randint(0, len(w) - 2)
                w = w[:i] + w[i+1] + w[i] + w[i+2:]
        elif r < 0.04:
            w = w.upper()
        elif r < 0.05:
            abbrevs = {'using': 'w/', 'and': '&', 'development': 'dev', 'management': 'mgmt'}
            w = abbrevs.get(w.lower(), w)
        result.append(w)
    return ' '.join(result)

def generate_synthetic_resume(category, inject=False):
    frags = FRAGMENTS[category]
    n = random.randint(4, 8)
    chosen = random.sample(frags, min(n, len(frags)))
    text = '. '.join(chosen) + '.'
    if inject:
        text = inject_noise(text)
    return text

print("=" * 50)
print("SKILLER – Resume Dataset Preparation")
print("=" * 50)

if os.path.exists(DATASET_PATH):
    print(f"\n[INFO] Kaggle dataset found at '{DATASET_PATH}'. Loading...")
    try:
        df_raw = pd.read_csv(DATASET_PATH)
        print(f"[INFO] Raw shape: {df_raw.shape}")
        print(f"[INFO] Columns: {list(df_raw.columns)}")

        # Auto-detect resume text column
        text_col = None
        for col in ['Resume_str', 'Resume', 'Resume_Text', 'resume']:
            if col in df_raw.columns:
                text_col = col
                break
        if not text_col:
            raise ValueError("Could not find resume text column.")
        print(f"[INFO] Using text column: '{text_col}'")

        # Auto-detect category column
        cat_col = None
        for col in ['Category', 'category', 'Role', 'role', 'Title', 'title']:
            if col in df_raw.columns:
                cat_col = col
                break
        if not cat_col:
            raise ValueError("Could not find category column.")
        print(f"[INFO] Using category column: '{cat_col}'")

        df = df_raw[[text_col, cat_col]].copy()
        df.columns = ['Resume', 'Category']
        df.dropna(inplace=True)
        df.drop_duplicates(subset='Resume', inplace=True)
        df = df[df['Resume'].str.len() >= 50]
        df['Resume'] = df['Resume'].str.strip()
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"[OK] Cleaned dataset saved to '{OUTPUT_PATH}' | Rows: {len(df)}")

    except Exception as e:
        print(f"[WARNING] Error processing Kaggle dataset: {e}")
        print("[INFO] Falling back to synthetic generation...")
        os.path.exists(DATASET_PATH) and os.remove(DATASET_PATH)
        df = None
else:
    print(f"\n[INFO] No Kaggle dataset found at '{DATASET_PATH}'.")
    df = None

if df is None or not os.path.exists(OUTPUT_PATH):
    print("\n[INFO] Generating synthetic dataset (2000 rows)...")
    rows = []
    per_category = 2000 // len(CATEGORIES)

    for cat in CATEGORIES:
        for i in range(per_category):
            inject = (i % 20 == 0)  # ~5% noise
            text = generate_synthetic_resume(cat, inject)
            rows.append({'Resume': text, 'Category': cat})

    # Fill remaining rows
    for i in range(2000 - len(rows)):
        cat = random.choice(CATEGORIES)
        rows.append({'Resume': generate_synthetic_resume(cat), 'Category': cat})

    df_syn = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    os.makedirs('../datasets', exist_ok=True)
    df_syn.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Synthetic dataset saved to '{OUTPUT_PATH}' | Rows: {len(df_syn)}")
    print(f"[INFO] Category distribution:\n{df_syn['Category'].value_counts()}")

print("\n[OK] Dataset preparation complete.")
