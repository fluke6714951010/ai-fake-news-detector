import streamlit as st
import joblib
import os
from utils.preprocess import clean_text

# =========================
# PATH CONFIG
# =========================
MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

# =========================
# LOAD MODEL FUNCTION
# =========================
@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Please run: python -m model.train_model")
        st.stop()

    if not os.path.exists(VECTORIZER_PATH):
        st.error("❌ Vectorizer not found. Please run: python -m model.train_model")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer

    except Exception as e:
        st.error("❌ Model corrupted. Please retrain model.")
        st.text(str(e))
        st.stop()

model, vectorizer = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Fake News Detector", layout="centered")

st.title("🧠 AI Fake News Detector")
st.write("Paste news text and let AI detect if it is Fake or Real")

# =========================
# INPUT
# =========================
news_text = st.text_area("📰 Enter News Text")

detect_button = st.button("🔍 Detect News")

# =========================
# PREDICTION
# =========================
if detect_button:

    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
        st.stop()

    cleaned = clean_text(news_text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector).max()

    st.divider()

    if prediction == 0:
        st.error("🚨 Fake News Detected")
    else:
        st.success("✅ Real News")

    st.write(f"Confidence: {probability:.2f}")