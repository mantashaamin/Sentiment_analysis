import joblib
from pathlib import Path

from preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_tfidf.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


# Load trained model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(review):

    # Apply the same preprocessing
    cleaned_review = clean_text(review)

    # Convert text into TF-IDF features
    review_vector = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = model.predict(review_vector)

    return prediction[0]


if __name__ == "__main__":

    print("\n===== SENTIMENT ANALYSIS SYSTEM =====")

    review = input("\nEnter your app review: ")

    result = predict_sentiment(review)

    print("\nOriginal Review:", review)
    print("Predicted Sentiment:", result)