
import streamlit as st
import joblib
from pathlib import Path
import sys

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(str(PROJECT_ROOT / "src"))

from preprocessing import clean_text

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_tfidf.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="centered"
)

st.title("Google Play Review Sentiment Analysis")
st.write("Analyze the sentiment of Google Play Store app reviews.")

# Review input
review = st.text_area(
    "Enter your app review:",
    placeholder="Example: This app is amazing and very useful..."
)

if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        cleaned_review = clean_text(review)

        if not cleaned_review:
            st.warning("Please enter a review containing usable text.")

        else:
            review_vector = vectorizer.transform([cleaned_review])

            prediction = model.predict(review_vector)[0]

            st.subheader("Prediction")

            if prediction == "Positive":
                st.success(f"Sentiment: {prediction}")
            elif prediction == "Negative":
                st.error(f"Sentiment: {prediction}")
            else:
                st.info(f"Sentiment: {prediction}")

            # Display probabilities
            probabilities = model.predict_proba(review_vector)[0]

            st.subheader("Sentiment Probabilities")

            for sentiment, probability in zip(
                model.classes_, probabilities
            ):
                st.write(f"{sentiment}: {probability:.2%}")
                st.progress(float(probability))
