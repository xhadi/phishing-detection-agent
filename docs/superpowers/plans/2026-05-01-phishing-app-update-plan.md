# Phishing App Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `app.py` to use the new XGBoost model, SentenceTransformer embeddings, and separate Email/SMS input fields.

**Architecture:** A Streamlit frontend that takes text inputs, vectorizes them with a pre-trained SentenceTransformer, and predicts using an XGBoost classifier. Explainability is handled via a custom Leave-One-Out word omission heuristic.

**Tech Stack:** Python, Streamlit, XGBoost, Sentence-Transformers, Pandas, scikit-learn (implied by model).

---

### Task 1: Update Model Loading and Initialization

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update Imports**
Replace vectorizer imports and add necessary libraries.
```python
import streamlit as st
import pickle
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
```

- [ ] **Step 2: Update `load_model` function**
Replace the old `load_model` logic to load the XGBoost model and initialize SentenceTransformer.
```python
@st.cache_resource
def load_model():
    with open('threat_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return model, embedder
```

- [ ] **Step 3: Update Model Instantiation Call**
Update the try-except block where the model is loaded in the main script.
```python
# Load model and embedder
try:
    model, embedder = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
```

- [ ] **Step 4: Commit**
```bash
git add app.py
git commit -m "refactor: update model loading logic to use XGBoost and SentenceTransformer"
```

### Task 2: Implement UI Input Logic (Email vs SMS)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace old `user_input` UI with Radio Button and Conditional Fields**
Find the `# GUI interface` section and replace the single `st.text_area` with type selection.
```python
# GUI interface
input_type = st.radio("Select Message Type:", ["Email", "SMS"])

user_input = ""
if input_type == "Email":
    sender = st.text_input("Sender (From):")
    subject = st.text_input("Subject:")
    body = st.text_area("Body:", height=200)
    if sender or subject or body:
        user_input = f"From: {sender} Subject: {subject} Body: {body}"
else:
    user_input = st.text_area("Enter the SMS text here:", height=200)
```

- [ ] **Step 2: Commit**
```bash
git add app.py
git commit -m "feat: add conditional UI fields for Email and SMS inputs"
```

### Task 3: Update Prediction and Results Display

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace Predict Logic**
Inside `if st.button("Analyze"):` block, update the vectorization and prediction logic.
```python
        # Preprocess and vectorize the input
        input_vector = embedder.encode([user_input])
        
        # Predict using the loaded model
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]
        
        # Map integer labels to string names
        label_map = {
            0: 'Safe Email',
            1: 'Safe SMS',
            2: 'Spam Email',
            3: 'Phishing Email',
            4: 'Malicious SMS'
        }
        predicted_class_name = label_map.get(prediction, "Unknown")
        predicted_probability = probabilities[prediction]
```

- [ ] **Step 2: Update Display Logic**
Update the success/error messages based on the 5 classes.
```python
        # Display results based on predicted class
        st.write(f"### Predicted Category: **{predicted_class_name}**")
        st.write(f"**Confidence:** {predicted_probability * 100:.2f}%")

        if prediction in [2, 3, 4]:  # Spam, Phishing, Malicious
            st.error(f"⚠️ This message is flagged as a THREAT: {predicted_class_name}.")
        else:  # Safe Email, Safe SMS
            st.success(f"✅ This message appears to be SAFE: {predicted_class_name}.")
```

- [ ] **Step 3: Commit**
```bash
git add app.py
git commit -m "feat: implement 5-class prediction mapping and alert displays"
```

### Task 4: Implement Leave-One-Out Explainability

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Remove Old Highlight Logic**
Remove the old `st.markdown` legend and `highlight_words` function.

- [ ] **Step 2: Implement Leave-One-Out Heuristic**
Under `# --- Explainability Feature: Highlighting Key Words ---`, add the new logic.
```python
        st.write("### Analysis Breakdown:")
        st.write("Highlighted words indicate their contribution to the predicted class. Red highlights push the model *towards* this prediction.")
        
        # Display the Legend
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-bottom: 20px; font-size: 14px;">
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="background-color: rgba(255, 0, 0, 0.8); width: 15px; height: 15px; border-radius: 3px;"></span> Strong Indicator
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="background-color: rgba(255, 0, 0, 0.3); width: 15px; height: 15px; border-radius: 3px;"></span> Weak Indicator
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        words = re.findall(r'\b\w+\b', user_input)
        unique_words = set(words)
        
        word_importance = {}
        for word in unique_words:
            # Create text without this word (using regex to remove whole word)
            text_without_word = re.sub(r'\b' + re.escape(word) + r'\b', '', user_input, flags=re.IGNORECASE)
            
            # Predict probability of the original predicted class WITHOUT this word
            vec_without = embedder.encode([text_without_word])
            prob_without = model.predict_proba(vec_without)[0][prediction]
            
            # Importance = Baseline Prob - Prob Without Word
            # If positive, the word INCREASES the probability (it's an indicator)
            importance = predicted_probability - prob_without
            word_importance[word.lower()] = importance

        def highlight_words_new(match):
            word = match.group(0)
            word_lower = word.lower()
            if word_lower in word_importance:
                imp = word_importance[word_lower]
                threshold = 0.05  # minimum 5% probability drop to highlight
                
                if imp > threshold:
                    # scale intensity up to a max drop of 0.3 (30%)
                    intensity = min(1.0, imp / 0.3)
                    return f'<span style="background-color: rgba(255, 0, 0, {intensity}); padding: 2px; border-radius: 4px; color: white;">{word}</span>'
            return word

        highlighted_text = re.sub(r'\b\w+\b', highlight_words_new, user_input)
        st.markdown(f"<div style='line-height:1.6;'>{highlighted_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
```

- [ ] **Step 3: Commit**
```bash
git add app.py
git commit -m "feat: add leave-one-out word highlighting explainability"
```
