import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Function to load the trained model and vectorizer
@st.cache_resource
def load_model():
    with open('trained_model/threat_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return model, embedder

# Page configuration
st.set_page_config(page_title="Phishing Detection Agent", page_icon="🔍", layout="wide")

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
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

def handle_analyze():
    st.session_state.analyzed = True

if st.session_state.analyzed:
    col1, col2 = st.columns(2, gap="large")
    input_container = col1
    result_container = col2
else:
    input_container = st.container()
    result_container = st.container()

with input_container:
    input_type = st.radio("Select Message Type:", ["Email", "SMS"])

    user_input = ""
    if input_type == "Email":
        sender = st.text_input("Sender (From):")
        subject = st.text_input("Subject:")
        body = st.text_area("Body:", height=200)
        if sender or subject or body:
                # Fill empty fields with 'nan' to match the training data
                sender = sender if sender else "nan"
                subject = subject if subject else "nan"
                
                # Exact spacing to match the training set concatenation
                user_input = f"From: {sender}  Subject: {subject}  Body: {body}"
    else:
        user_input = st.text_area("Enter the SMS text here:", height=200)

    analyze_button = st.button("Analyze", on_click=handle_analyze)

with result_container:
    if analyze_button:
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            # --- 1. EXTRACT STRUCTURAL FEATURES ---
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            urgency_keywords = ['urgent', 'suspend', 'decline', 'hold', 'alert', 'verify', 'update', 'immediate', 'restricted', 'validate', 'restore']
            urgency_pattern = '|'.join([rf'\b{word}\b' for word in urgency_keywords])
            
            contains_url = 1 if re.search(url_pattern, user_input) else 0
            msg_length = len(user_input)
            is_urgent = 1 if re.search(urgency_pattern, user_input, re.IGNORECASE) else 0

            # --- 2. CLEAN TEXT & GENERATE SEMANTIC VECTORS ---
            # Strip the URL out of the user's input to remove visual noise
            clean_user_input = re.sub(url_pattern, ' [URL] ', user_input)
            
            # Pass the CLEANED text to the embedder
            semantic_vector = embedder.encode([clean_user_input])
            structural_features = np.array([[contains_url, msg_length, is_urgent]])
            
            # Combine into the 387-dimensional array
            fused_input = np.hstack((semantic_vector, structural_features))
            
            # --- 3. PREDICT & ROUTE ---
            raw_prediction = int(model.predict(fused_input)[0])
            probabilities = model.predict_proba(fused_input)[0]
            
            prediction = raw_prediction
            # Application Layer Routing (UI Enforcement)
            if input_type == "SMS":
                if prediction == 2: prediction = 3
                elif prediction == 0: prediction = 1
            elif input_type == "Email":
                if prediction == 3: prediction = 2
                elif prediction == 1: prediction = 0
                    
            # Map integer labels to string names
            label_map = {
                0: 'Safe Email',
                1: 'Safe SMS',
                2: 'Phishing Email',
                3: 'Malicious SMS'
            }
            predicted_class_name = label_map.get(prediction, "Unknown")
            
            # Calculate combined confidence score based on Threat vs Safe
            if prediction in [2, 3]:
                predicted_probability = probabilities[2] + probabilities[3]
            else:
                predicted_probability = probabilities[0] + probabilities[1]
            
            # --- 4. DISPLAY RESULTS & HITL LOGIC ---
            st.write(f"### Predicted Category: **{predicted_class_name}**")
            st.write(f"**Confidence:** {predicted_probability * 100:.2f}%")

            # Human-in-the-Loop (HITL) Threshold Trigger
            if predicted_probability < 0.75:
                st.warning("⚠️ **THREAT UNKNOWN:** Confidence is below 75%. Sending to Security Analyst for Manual Review.")
            elif prediction in [2, 3]:  
                st.error(f"🚨 This message is flagged as a THREAT: {predicted_class_name}.")
            else:  
                st.success(f"✅ This message appears to be SAFE: {predicted_class_name}.")

            # --- 5. EXPLAINABILITY FEATURE (XAI) ---
            st.write("### Analysis Breakdown:")
            st.write("Highlighted words indicate their contribution to the predicted class. Red highlights push the model *towards* this prediction.")
            
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
                # Create text without this word
                text_without_word = re.sub(r'\b' + re.escape(word) + r'\b', '', user_input, flags=re.IGNORECASE)
                
                # Recalculate structural features
                c_url_wo = 1 if re.search(url_pattern, text_without_word) else 0
                m_len_wo = len(text_without_word)
                i_urg_wo = 1 if re.search(urgency_pattern, text_without_word, re.IGNORECASE) else 0
                
                # Re-embed and fuse (Clean the text of URLs here as well)
                clean_text_without_word = re.sub(url_pattern, ' [URL] ', text_without_word)
                vec_without = embedder.encode([clean_text_without_word])
                struct_without = np.array([[c_url_wo, m_len_wo, i_urg_wo]])
                fused_without = np.hstack((vec_without, struct_without))
                
                # Predict probability WITHOUT this word
                probs_without = model.predict_proba(fused_without)[0]
                
                # Calculate combined probability to match our routing logic
                if prediction in [2, 3]:
                    prob_without = probs_without[2] + probs_without[3]
                else:
                    prob_without = probs_without[0] + probs_without[1]
                
                # Importance = Baseline Prob - Prob Without Word
                importance = predicted_probability - prob_without
                word_importance[word.lower()] = importance

            def highlight_words_new(match):
                word = match.group(0)
                word_lower = word.lower()
                if word_lower in word_importance:
                    imp = word_importance[word_lower]
                    threshold = 0.005  
                    
                    if imp > threshold:
                        # scale intensity up to a max drop of 0.35
                        intensity = min(1.0, imp / 0.35)
                        return f'<span style="background-color: rgba(255, 0, 0, {intensity}); padding: 2px; border-radius: 4px; color: white;">{word}</span>'
                return word

            highlighted_text = re.sub(r'\b\w+\b', highlight_words_new, user_input)
            st.markdown(f"<div style='line-height:1.6;'>{highlighted_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)