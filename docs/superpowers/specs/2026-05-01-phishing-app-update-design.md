# Phishing App Update Design Spec

## Overview
Update the existing Streamlit phishing detection app (`app.py`) to align with the newly trained XGBoost model and SentenceTransformer embedding approach. The new model categorizes messages into 5 classes (Safe Email, Safe SMS, Spam Email, Phishing Email, Malicious SMS).

## Architecture
- **Frontend**: Streamlit web application.
- **Model Storage**: Pre-trained `threat_model_xgboost.pkl` loaded via `pickle`.
- **Feature Extraction**: `SentenceTransformer('all-MiniLM-L6-v2')` loaded via `sentence-transformers` library (cached with `@st.cache_resource`).
- **Dependencies**: `streamlit`, `pandas`, `xgboost`, `sentence-transformers`. (User will install these manually, no `requirements.txt` update required).

## Components & UI
1. **Model Loader**:
   - `load_model()` function will now load the `.pkl` XGBoost model and initialize `SentenceTransformer`. It will no longer load `vectorizer.pkl`.
   
2. **Input Interface**:
   - A `st.radio` or `st.selectbox` for selecting message type: **Email** vs **SMS**.
   - If **Email**: Render text inputs for `Sender`, `Subject`, and `Body`.
   - If **SMS**: Render a single text area for the message.

3. **Data Preprocessing**:
   - Concatenate inputs identically to the training logic:
     - Email: `"From: " + sender + " Subject: " + subject + " Body: " + body`
     - SMS: Raw text.

4. **Prediction & Display**:
   - Obtain embedding via `embedder.encode([text_combined])`.
   - Get prediction and probabilities from XGBoost model (`model.predict`, `model.predict_proba`).
   - Map integer prediction to class name:
     - 0: Safe Email
     - 1: Safe SMS
     - 2: Spam Email
     - 3: Phishing Email
     - 4: Malicious SMS
   - Display an appropriate alert (`st.error`, `st.warning`, `st.success`) based on severity, alongside the probability.

5. **Explainability (Word Highlighting)**:
   - Replace linear coefficients (`model.coef_`) with a "Leave-One-Out" heuristic:
     - Given the base probability of the predicted class ($P_{base}$).
     - Split text into unique words/tokens.
     - For each word, remove it from the text, get the new embedding, and predict the probability of the same class ($P_{without}$).
     - The importance of the word is $P_{base} - P_{without}$.
     - Significant drops indicate high importance.
   - Display the text with HTML `<span>` tags to dynamically color words based on their importance to the *predicted* class (intensity mapped from difference). Red/orange for threats, green for safe classes.

## Error Handling & Edge Cases
- Empty inputs gracefully handled with warnings.
- `try-except` blocks for model loading, displaying user-friendly errors if dependencies (`sentence-transformers`, `xgboost`) are missing.

## Testing Strategy
- Manual verification of Streamlit UI components (Email vs SMS).
- Verification of word highlighting using a known phishing string to see if suspicious words light up.
