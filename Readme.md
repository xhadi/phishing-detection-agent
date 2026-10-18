# 🔍 Phishing Detection Agent (Hybrid Architecture)

The **Phishing Detection Agent** is an Threat Intelligence tool designed to intercept and classify modern cyber threats across both Email and SMS channels. 

Moving beyond basic keyword matching, this agent utilizes a **Hybrid AI Architecture**—combining dense semantic vector embeddings with explicit structural feature engineering to detect psychological manipulation, credential harvesting, and MFA bypass attempts. 

## ✨ Core Architecture & Features
- **Hybrid XGBoost Engine:** Fuses 384-dimensional semantic text vectors (via `all-MiniLM-L6-v2`) with structural features (urgency vocabulary; for SMS also link presence and count) to identify sophisticated threats.
- **Channel-Specific Models:** One binary model per channel, selected by the message type in the UI: `Safe Email` / `Phishing Email` and `Safe SMS` / `Malicious SMS`.
- **Shared Preprocessing:** `phishing_features.py` holds all text cleaning and feature code, used by both the training notebook and the app, so the model always sees input prepared the same way.
- **Human-in-the-Loop (HITL):** Automatically detects edge cases. If threat confidence falls below 75%, the system intercepts the message and flags it for manual analyst review.
- **Explainable AI (XAI):** The interface mathematically calculates and highlights the exact words that triggered the threat detection, providing transparency into the model's psychological analysis. 

## 🛠️ Technology Stack
- **Machine Learning:** `XGBoost` (Classification) and `Scikit-Learn` (Metrics/Weighting).
- **Natural Language Processing (NLP):** `sentence-transformers` (all-MiniLM-L6-v2) for contextual text embeddings.
- **Data Engineering:** `Pandas` and `NumPy` for merging, cleaning and deduplicating the CSV sources.
- **Frontend/UI:** `Streamlit` for the dark-themed, interactive analysis portal.

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required data and ML packages:
```bash
pip install -r requirements.txt
```

If you prefer a single command instead of a requirements file, run:
```bash
pip install pandas numpy scikit-learn xgboost sentence-transformers streamlit matplotlib
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

### 2. Run the Application
Once the dependencies are installed, you can start the Streamlit interface by running:
```bash
streamlit run app.py
```

### 3. Model Training
The app needs `trained_model/email_model.json` and `trained_model/sms_model.json`. To train them:
1. Download the datasets into `dataset/emails/` and `dataset/sms/` (sources and file list in [dataset/README.md](dataset/README.md)).
2. Upload the whole project folder to Google Drive as `My Drive/phishing-detection-agent`.
3. Open `train_model.ipynb` in Google Colab, select **Runtime → Change runtime type → T4 GPU**, then **Run all**.
4. Download `email_model.json`, `sms_model.json` and `model_info.json` from `trained_model/` in Drive into your local `trained_model/` folder.

The notebook also runs locally with Jupyter, but the embedding step is slow without a GPU.

Check the **"Reality check: hand-written modern messages"** section of the notebook after training. It scores the models on modern phishing examples that are not in the training data, which is a better guide than the test-set accuracy.

## 📁 Project Structure
```text
.
├── app.py                      # Main Streamlit web application
├── phishing_features.py        # Text cleaning + features shared by app and notebook
├── train_model.ipynb           # Notebook for training the models (Colab-ready)
├── trained_model/
│   ├── email_model.json        # Email model (XGBoost native format)
│   ├── sms_model.json          # SMS model
│   └── model_info.json         # Embedder, features and metrics of the trained models
├── dataset/                    # Training data (CSVs not committed, see dataset/README.md)
│   ├── emails/                 # Enron, Nazario, Nigerian_Fraud, SpamAssasin
│   ├── sms/                    # phishing_sms, SmishTank_dataset, spam_sms (+ unused sms_from_LLMs)
│   └── legacy/                 # Old v1 data (old_datasets_v1.zip)
├── requirements.txt            # Python dependencies
└── Readme.md                   # This documentation
```