import html
import json
import os
import re

import streamlit as st
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

from phishing_features import (
    EMBEDDER_NAME, STRUCTURAL_FEATURE_NAMES, build_email_text, embedding_dimension, riskiest_windows,
    threat_probabilities,
)

MODEL_DIR = os.environ.get('PHISHING_MODEL_DIR', 'trained_model')
CHANNELS = {'Email': 'email', 'SMS': 'sms'}  # UI label -> channel name used by phishing_features
CLASS_NAMES = {'Email': ('Safe Email', 'Phishing Email'), 'SMS': ('Safe SMS', 'Malicious SMS')}
REVIEW_THRESHOLD = 0.75


def load_model_info():
    """Read the settings the notebook saved and check they match phishing_features.py."""
    path = os.path.join(MODEL_DIR, 'model_info.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Download it with the models from train_model.ipynb.")
    with open(path) as f:
        info = json.load(f)
    if info['embedder'] != EMBEDDER_NAME:
        raise ValueError(
            f"The models were trained with the '{info['embedder']}' embedder but phishing_features.py uses "
            f"'{EMBEDDER_NAME}'. Retrain with train_model.ipynb."
        )
    for channel in CHANNELS.values():
        if info[channel]['structural_features'] != STRUCTURAL_FEATURE_NAMES[channel]:
            raise ValueError(
                f"The {channel} model was trained with structural features {info[channel]['structural_features']} "
                f"but phishing_features.py builds {STRUCTURAL_FEATURE_NAMES[channel]}. Retrain with train_model.ipynb."
            )
    return info


# Function to load the trained models and embedder
@st.cache_resource
def load_models():
    info = load_model_info()
    embedder = SentenceTransformer(EMBEDDER_NAME)
    models = {}
    for channel in CHANNELS.values():
        path = os.path.join(MODEL_DIR, f'{channel}_model.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' not found. Train the models with train_model.ipynb first.")
        model = XGBClassifier()
        model.load_model(path)
        expected_features = embedding_dimension(embedder) + len(STRUCTURAL_FEATURE_NAMES[channel])
        if model.n_features_in_ != expected_features:
            raise ValueError(
                f'{channel} model expects {model.n_features_in_} features but phishing_features.py builds '
                f'{expected_features}. Retrain with train_model.ipynb.'
            )
        models[channel] = model
    return models, embedder, info


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

# Load models and embedder
try:
    models, embedder, model_info = load_models()
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

    if input_type == "Email":
        subject = st.text_input("Subject:")
        body = st.text_area("Body:", height=200)
        user_input = build_email_text(subject, body)
    else:
        user_input = st.text_area("Enter the SMS text here:", height=200)

    analyze_button = st.button("Analyze", on_click=handle_analyze)

with result_container:
    if analyze_button:
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            channel = CHANNELS[input_type]
            model = models[channel]
            safe_name, threat_name = CLASS_NAMES[input_type]

            # --- 1. PREDICT ---
            threat_prob = float(threat_probabilities(model, embedder, [user_input], channel)[0])
            scored_text = user_input  # the text the word highlighting explains

            # Optional window scan: one suspicious passage can flag a long message (off when the threshold is None)
            window_threshold = model_info[channel].get('window_threshold')
            flagged_window = None
            if window_threshold is not None:
                window_probs, windows = riskiest_windows(model, embedder, [user_input], channel)
                if window_probs[0] >= window_threshold and window_probs[0] > threat_prob:
                    threat_prob, flagged_window = float(window_probs[0]), windows[0]
                    scored_text = flagged_window

            is_threat = threat_prob >= model_info['threshold']
            predicted_class_name = threat_name if is_threat else safe_name
            predicted_probability = threat_prob if is_threat else 1 - threat_prob

            # --- 2. DISPLAY RESULTS & HITL LOGIC ---
            st.write(f"### Predicted Category: **{predicted_class_name}**")
            st.write(f"**Confidence:** {predicted_probability * 100:.2f}%")

            # Human-in-the-Loop (HITL) Threshold Trigger
            if predicted_probability < REVIEW_THRESHOLD:
                st.warning("⚠️ **THREAT UNKNOWN:** Confidence is below 75%. Sending to Security Analyst for Manual Review.")
            elif is_threat:
                st.error(f"🚨 This message is flagged as a THREAT: {predicted_class_name}.")
            else:
                st.success(f"✅ This message appears to be SAFE: {predicted_class_name}.")
            if flagged_window:
                # Windows are cleaned text (lowercase words and numbers only), so this can't inject markdown
                st.info(f"Flagged because of this passage: \"{flagged_window}\"")

            # --- 3. EXPLAINABILITY FEATURE (XAI) ---
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

            # Importance = how much the predicted class's probability drops when the word is removed
            # from the scored text (the whole message, or the flagged passage).
            # All variants are scored in one batch instead of one embedding call per word.
            unique_words = sorted({w.lower() for w in re.findall(r'\b\w+\b', scored_text)})
            variants = [
                re.sub(r'\b' + re.escape(word) + r'\b', '', scored_text, flags=re.IGNORECASE)
                for word in unique_words
            ]
            with st.spinner("Calculating word contributions..."):
                probs_without = threat_probabilities(model, embedder, variants, channel) if variants else []

            word_importance = {}
            for word, prob_without in zip(unique_words, probs_without):
                class_prob_without = prob_without if is_threat else 1 - prob_without
                word_importance[word] = predicted_probability - class_prob_without

            def highlight_word(word):
                imp = word_importance.get(word.lower(), 0)
                threshold = 0.005
                if imp > threshold:
                    # scale intensity up to a max drop of 0.35
                    intensity = min(1.0, imp / 0.35)
                    return f'<span style="background-color: rgba(255, 0, 0, {intensity}); padding: 2px; border-radius: 4px; color: white;">{word}</span>'
                return word

            # Odd indices of the split are words; everything else is escaped so user input can't inject HTML
            parts = re.split(r'(\w+)', user_input)
            highlighted_text = ''.join(highlight_word(p) if i % 2 else html.escape(p) for i, p in enumerate(parts))
            st.markdown(f"<div style='line-height:1.6;'>{highlighted_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
