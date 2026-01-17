# 🍽️ Zomato Review Sentiment Analysis

🔗 **Live App:**  
https://zomato-sentiment-analysis-2t6wvbfbxgpdym3suhidz6.streamlit.app/

📓 **Notebook:**  
`Sentiment_Analysis_Zomoto_reviews.ipynb`

---

## 📌 Overview

This project performs **sentiment analysis on Zomato restaurant reviews** using a **Bi-Directional GRU deep learning model**.  
Along with predicting **Positive or Negative sentiment**, the application explains predictions using **LIME (Explainable AI)**.

The project demonstrates an **end-to-end NLP workflow** from model development to production deployment.

---

## 🚀 Features

- Bi-GRU deep learning model for contextual understanding  
- Handles class imbalance using class weights  
- Precision-focused evaluation strategy  
- Optimal probability threshold (**0.69**) selected using ROC analysis  
- Word-level explainability using **LIME**  
- Clean, dark-theme friendly **Streamlit web app**  
- Fully deployed and accessible online  

---

## 📊 Model Evaluation

- **Precision:** ~0.98  
- **Recall:** ~0.93  
- **F1-Score:** ~0.95  

**Why precision?**  
False positives (negative reviews predicted as positive) are costly in real-world review systems, so precision was prioritized.

---

## 🔍 Explainability (LIME)

LIME highlights the **words that influence each prediction**:
- 🟢 Positive words push the prediction toward *Positive*
- 🔴 Negative words push the prediction toward *Negative*

This improves transparency and trust in model decisions.

---

## 🧪 Example Reviews

**Positive** : The food was delicious and the staff were very polite.
**Negative** : The service was terrible and the food was cold.


---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- Streamlit  
- scikit-learn  
- LIME  
- NumPy, joblib, emoji  

---

## 📦 Run Locally

```bash
git clone https://github.com/hemanthk24/Zomato-Sentiment-Analysis.git
cd Zomato-Sentiment-Analysis
pip install -r requirements.txt
streamlit run app.py

## ⚠️ Limitations

- Sarcasm detection is limited  
- Aspect-level sentiment (food vs service) is not separated  
- Very short reviews may not produce explanations  

## 🔮 Future Improvements

- Aspect-based sentiment analysis  
- Transformer-based models (BERT)  
- Multilingual sentiment support  

## 👤 Author

**Hemanth Goud**  
Aspiring Machine Learning
Interested in **NLP, Deep Learning, and Explainable AI**

