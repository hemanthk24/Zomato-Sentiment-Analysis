import streamlit as st
import joblib
import re
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer
import emoji

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Zomato Review Sentiment",
    page_icon="🍽️",
    layout="centered"
)

THRESHOLD = 0.69
MAXLEN = 100

# ---------------- LOAD MODEL ----------------
model = load_model("sentiment_bi_gru_model.keras")
tokenizer = joblib.load("tokenizer.joblib")

# ---------------- CSS ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
}

/* Title */
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #9aa0a6;
    margin-bottom: 30px;
}

/* Prediction Card (dark-theme friendly) */
.pred-card {
    padding: 20px;
    border-radius: 14px;
    background-color: rgba(255, 255, 255, 0.08);
    border-left: 6px solid #4CAF50;
    color: #f1f3f4;
    margin-top: 15px;
}

.neg {
    border-left-color: #e74c3c;
}

/* LIME box */
.lime-box {
    background-color: rgba(255, 255, 255, 0.06);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #f1f3f4;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI HEADER ----------------
st.markdown('<div class="main-title">🍽️ Zomato Review Sentiment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Bi-GRU Deep Learning Model with LIME Explainability</div>',
    unsafe_allow_html=True
)

# ---------------- PREPROCESS ----------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------------- MODEL PREDICTION ----------------
def predict_proba(texts):
    safe_texts = []

    for t in texts:
        t = preprocess_text(t)
        if t.strip() == "":
            t = "neutral"
        safe_texts.append(t)

    sequences = tokenizer.texts_to_sequences(safe_texts)
    padded = pad_sequences(sequences, maxlen=MAXLEN, padding="post", truncating="post")

    probs = model.predict(padded)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

    return np.hstack([1 - probs, probs])

def predict_sentiment(text):
    prob = predict_proba([text])[0][1]
    label = "Positive" if prob >= THRESHOLD else "Negative"
    return label, prob

# ---------------- LIME ----------------
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

def explain_prediction(text):
    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=6,
        num_samples=1000
    )
    return explanation.as_list()

# ---------------- INPUT ----------------
st.subheader("✍️ Enter a review")

user_input = st.text_area(
    "",
    height=130,
    placeholder="I was happy with food but the staff were not good"
)

analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ---------------- OUTPUT ----------------
if analyze:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, probability = predict_sentiment(user_input)
        card_class = "pred-card" if sentiment == "Positive" else "pred-card neg"

        st.markdown(f"""
        <div class="{card_class}">
            <h3>📌 Prediction</h3>
            <p><b>Sentiment:</b> {sentiment}</p>
            <p><b>Confidence:</b> {probability:.2%}</p>
            <p><b>Decision Threshold:</b> {THRESHOLD}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Explanation (LIME)")

        if len(user_input.split()) < 3:
            st.info("Explanation is skipped for very short inputs.")
        else:
            explanation = explain_prediction(user_input)

            st.markdown('<div class="lime-box">', unsafe_allow_html=True)
            for word, weight in explanation:
                if weight > 0:
                    st.markdown(f"🟢 **{word}** → +{weight:.3f}")
                else:
                    st.markdown(f"🔴 **{word}** → {weight:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.caption(
            "🟢 Positive weights push prediction toward Positive sentiment • "
            "🔴 Negative weights push toward Negative sentiment"
        )
