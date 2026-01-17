import streamlit as st
import joblib
import re
from tensorflow import keras
from keras.models import load_model
import numpy as np
from lime.lime_text import LimeTextExplainer
from keras.preprocessing.sequence import pad_sequences
import emoji

# Load the pre-trained model
model = load_model('sentiment_bi_gru_model.keras')
tokenizer = joblib.load('tokenizer.joblib')

# Define threshold and Maxlen for classification
THRESHOLD = 0.69
MAXLEN = 100


# Function to preprocess input text
def preprocess_text(text):
    # Convert to string (safety)
    text = str(text)  
    # Lowercase
    text = text.lower() 
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)   
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)  
    # Convert emojis to words
    text = emoji.demojize(text, delimiters=(" ", " "))   
    # Remove non-alphabet characters
    text = re.sub(r"[^a-z\s]", "", text)   
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- MODEL PREDICTION ----------------
def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=MAXLEN,
        padding="post",
        truncating="post"
    )
    probs = model.predict(padded)
    return np.hstack([1 - probs, probs])

def predict_sentiment(text):
    processed = preprocess_text(text)
    prob = predict_proba([processed])[0][1]
    label = "Positive" if prob >= THRESHOLD else "Negative"
    return label, prob

# ---------------- LIME EXPLAINER ----------------
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

def explain_prediction(text):
    explanation = explainer.explain_instance(
        text,
        predict_proba,
        num_features=8
    )
    return explanation.as_list()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Restaurant Review Sentiment",
    page_icon="🍽️",
    layout="centered"
)

st.markdown("""
<style>
/* Remove extra top padding */
.block-container {
    padding-top: 3rem;
}

/* Center title */
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;
}

/* Subtitle styling */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #6c757d;
    margin-bottom: 30px;
}

/* Prediction card */
.pred-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f8f9fa;
    border-left: 6px solid #4CAF50;
    margin-top: 15px;
}

/* Negative border */
.neg {
    border-left-color: #e74c3c;
}

/* LIME section */
.lime-box {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🍽️ Zomato Review Sentiment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Bi-GRU Deep Learning Model with LIME Explainability</div>',
    unsafe_allow_html=True
)


st.subheader("✍️ Enter a review")

user_input = st.text_area(
    "",
    height=130,
    placeholder="I was happy with food but the staff were not good"
)

analyze = st.button("🔍 Analyze Sentiment", use_container_width=True)

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
