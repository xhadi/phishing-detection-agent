import streamlit as st
import pickle
import re
import pandas as pd
from sentence_transformers import SentenceTransformer

# function to load the trained model and vectorizer
@st.cache_resource
def load_model():
    with open('threat_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return model, embedder

# Page configuration
st.set_page_config(page_title="Phishing Detection Agent", page_icon="🔍")

# Custom CSS for dark blue background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0a192f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Phishing Detection Agent")
st.write("Enter a message or email below to analyze its probability of being a phishing attempt.")

# Load model and embedder
try:
    model, embedder = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

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
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and vectorize the input
        input_vector = vectorizer.transform([user_input])
        
        # Predict using the loaded model
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0][1]  # Probability of being phishing
        
        # Display results
        if prediction == 1:
            st.error(f"⚠️ This message is likely a PHISHING attempt! (Probability: {probability * 100:.2f}%)")
        else:
            st.success(f"✅ This message appears to be LEGITIMATE. (Probability of phishing: {probability * 100:.2f}%)")

        # --- Explainability Feature: Highlighting Key Words ---
        st.write("### Analysis Breakdown:")
        st.write("Highlighted words indicate their contribution to the model's decision.")
        
        # Display the Legend
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-bottom: 20px; font-size: 14px;">
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="background-color: rgba(255, 0, 0, 1.0); width: 15px; height: 15px; border-radius: 3px;"></span> Strong Phishing
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="background-color: rgba(255, 0, 0, 0.4); width: 15px; height: 15px; border-radius: 3px;"></span> Weak Phishing
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="background-color: rgba(0, 255, 0, 1.0); width: 15px; height: 15px; border-radius: 3px;"></span> Strong Legitimate
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <span style="background-color: rgba(0, 255, 0, 0.4); width: 15px; height: 15px; border-radius: 3px;"></span> Weak Legitimate
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # The coefficients mapped to feature indices
        coefs = model.coef_[0]
        vocab = vectorizer.vocabulary_
        
        # Function to highlight words
        def highlight_words(match):
            word = match.group(0)
            word_lower = word.lower()
            if word_lower in vocab:
                idx = vocab[word_lower]
                weight = coefs[idx]
                
                # Highlight if weight strongly pushes to Phishing (positive) or Legitimate (negative)
                # You can adjust the threshold (e.g., 0.2) based on how much you want to highlight
                threshold = 0.2
                if weight > threshold:
                    intensity = min(1.0, weight / 4.0)  # scale for color intensity
                    return f'<span style="background-color: rgba(255, 0, 0, {intensity}); padding: 2px; border-radius: 4px; color: white;">{word}</span>'
                elif weight < -threshold:
                    intensity = min(1.0, abs(weight) / 4.0)
                    return f'<span style="background-color: rgba(0, 255, 0, {intensity}); padding: 2px; border-radius: 4px; color: black;">{word}</span>'
            return word

        # Replace words in original text while preserving structure
        highlighted_text = re.sub(r'\b\w+\b', highlight_words, user_input)
        
        # Display the custom HTML 
        # (Replacing newlines with HTML line breaks to maintain email formatting)
        st.markdown(f"<div style='line-height:1.6;'>{highlighted_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)