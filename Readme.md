# 🔍 Enterprise Phishing Detection Agent (Hybrid Architecture)

The **Phishing Detection Agent** is an enterprise-grade Threat Intelligence tool designed to intercept and classify modern cyber threats across both Email and SMS channels. 

Moving beyond basic keyword matching, this agent utilizes a **Hybrid AI Architecture**—combining dense semantic vector embeddings with explicit structural feature engineering to detect psychological manipulation, credential harvesting, and MFA bypass attempts. 

## ✨ Core Architecture & Features
- **Hybrid XGBoost Engine:** Fuses 384-dimensional semantic text vectors (via `all-MiniLM-L6-v2`) with custom structural logic (URL presence, message length, and urgency vocabulary) to identify sophisticated threats.
- **4-Class Threat Matrix:** Streamlined for enterprise Security Operations Centers (SOC). The model classifies inputs strictly into: `Safe Email`, `Safe SMS`, `Phishing Email`, and `Malicious SMS`.
- **Application-Layer Routing:** Enforces logical UI constraints on top of machine learning outputs to ensure the predicted threat matches the user's selected delivery channel.
- **Human-in-the-Loop (HITL):** Automatically detects edge cases. If threat confidence falls below 75%, the system intercepts the message and flags it for manual analyst review.
- **Explainable AI (XAI):** The interface mathematically calculates and highlights the exact words that triggered the threat detection, providing transparency into the model's psychological analysis. 

## 🛠️ Technology Stack
- **Machine Learning:** `XGBoost` (Classification) and `Scikit-Learn` (Metrics/Weighting).
- **Natural Language Processing (NLP):** `sentence-transformers` (all-MiniLM-L6-v2) for contextual text embeddings.
- **Data Engineering:** `Pandas`, `NumPy`, and `PyArrow` for processing large-scale Parquet files and CSV fusions.
- **Frontend/UI:** `Streamlit` for the dark-themed, interactive analysis portal.

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required data and ML packages:
```bash
pip install -r requirements.txt
```

If you prefer a single command instead of a requirements file, run:
```bash
pip install pandas numpy scikit-learn xgboost sentence-transformers streamlit matplotlib pyarrow
```

It's recommended to create and activate a virtual environment before installing dependencies.

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, run:
```bash
pip install -r requirements.txt
```