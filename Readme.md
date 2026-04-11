# 🔍 Phishing Detection Agent

The **Phishing Detection Agent** is a machine learning-based tool designed to analyze email and message content to predict whether it is a phishing attempt or a legitimate message. It includes an interactive web interface that provides transparency on the model's decision-making process.

## ✨ Features
- **High Accuracy Classification:** Trained using Scikit-Learn to detect malicious patterns in text data.
- **Visual Explainability (XAI):** The web app highlights specific words in your text that influenced the model's prediction:
  - **Red Highlight:** Words associated with Phishing attempts.
  - **Green Highlight:** Words associated with Legitimate correspondence.
  - Includes a legend denoting the strength (transparency/intensity) of the word's influence.
- **Open Source Data:** Includes free and open-source datasets (`phishing_email.csv` and `phishing_sms.csv`) used to train the model, making it reproducible for everyone.

## 🛠️ Technology Stack
- **Python 3.x**
- **Scikit-Learn:** Text vectorization (`TfidfVectorizer`) and ML prediction (`LogisticRegression`).
- **Pandas & Matplotlib:** Data loading, manipulation, and visualization (e.g., Confusion Matrix).
- **Streamlit:** Interactive, dark-themed web interface.

## 🚀 Getting Started

### 1. Install Dependencies
Make sure you have Python installed. Then, install the required packages:
```bash
pip install pandas scikit-learn matplotlib streamlit
```

### 2. Train the Model
The repository includes a Jupyter Notebook ready for you to train the ML pipeline from scratch. 
1. Open `train_model.ipynb`.
2. Run all the cells. This will process the `phishing_email.csv` and `phishing_sms.csv` files, train the model, display evaluation metrics, and generate two required files:
   - `phishing_model.pkl` (The trained Logistic Regression model)
   - `vectorizer.pkl` (The TF-IDF vectorizer)

### 3. Run the Detection App
Once the model is trained and saved, you can spin up the Streamlit web application by running:
```bash
streamlit run app.py
```
This will open the Phishing Detection Agent in your default web browser where you can paste emails to be analyzed!

## 📂 Project Structure
- `app.py`: The Streamlit web application script.
- `train_model.ipynb`: Jupyter Notebook for training and evaluating the ML model.
- `phishing_email.csv`: The open-source email dataset used for training.
- `phishing_sms.csv`: The open-source SMS dataset used for training.