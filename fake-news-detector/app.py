# app.py
import re
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("fake_news_detector.joblib")

def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.caption("TF-IDF + linear model (Logistic Regression or Passive Aggressive — chosen by CV)")

txt = st.text_area("Paste article text here", height=240, help="The text is cleaned (lowercased, URLs removed) before prediction.")
model = load_model()

col1, col2 = st.columns([1,1])
with col1:
    predict_btn = st.button("Predict")
with col2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

if predict_btn and txt.strip():
    cleaned = clean_text(txt)
    pred = model.predict([cleaned])[0]
    st.success(f"Prediction: **{pred}**")

    # Show probability/confidence if available
    clf = model.named_steps.get("clf", None)
    if clf is not None and hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(model.named_steps["tfidf"].transform([cleaned]))[0]
        classes = getattr(clf, "classes_", None)
        if classes is not None:
            st.write("Confidence:")
            for c, p in zip(classes, proba):
                st.write(f"- {c}: {p:.3f}")
