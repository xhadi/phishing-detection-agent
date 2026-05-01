import streamlit as st
import pickle
import re
import pandas as pd
from sentence_transformers import SentenceTransformer

# function to load the trained model and vectorizer
@st.cache_resource
def load_model():
    with open('trained_model/threat_model_xgboost.pkl', 'rb') as f:
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
        # Note the exact spacing to match the training set concatenation
        user_input = f"From: {sender}  Subject: {subject}  Body: {body}"
else:
    user_input = st.text_area("Enter the SMS text here:", height=200)
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and vectorize the input
        input_vector = embedder.encode([user_input])
        
        # Predict using the loaded model
        prediction = int(model.predict(input_vector)[0])
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
        
        # Display results based on predicted class
        st.write(f"### Predicted Category: **{predicted_class_name}**")
        st.write(f"**Confidence:** {predicted_probability * 100:.2f}%")

        if prediction in [2, 3, 4]:  # Spam, Phishing, Malicious
            st.error(f"⚠️ This message is flagged as a THREAT: {predicted_class_name}.")
        else:  # Safe Email, Safe SMS
            st.success(f"✅ This message appears to be SAFE: {predicted_class_name}.")

        # --- Explainability Feature: Highlighting Key Words ---
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
                # Lower threshold for embeddings, which distribute weights more broadly
                threshold = 0.01  
                
                if imp > threshold:
                    # scale intensity up to a max drop of 0.15 (15%)
                    intensity = min(1.0, imp / 0.15)
                    return f'<span style="background-color: rgba(255, 0, 0, {intensity}); padding: 2px; border-radius: 4px; color: white;">{word}</span>'
            return word

        highlighted_text = re.sub(r'\b\w+\b', highlight_words_new, user_input)
        st.markdown(f"<div style='line-height:1.6;'>{highlighted_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)