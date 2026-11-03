import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "dataset" / "processed" / "cleaned_reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "processed" / "preprocessed_reviews.csv"


# Download stopwords if they are not already available
nltk.download("stopwords", quiet=True)

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Words that must be preserved for sentiment analysis
# Important words that should not be removed
important_words = {
    # Negation words
    "no", "not", "nor", "never", "neither", "none",
    "cannot", "dont", "doesnt", "didnt", "isnt",
    "wasnt", "werent", "shouldnt", "wouldnt",
    "couldnt", "wont", "mustnt",

    # Contrast words
    "but", "however", "although", "though", "yet",

    # Intensity and emphasis words
    "very", "too", "really", "extremely", "quite",
    "highly", "most", "more", "less", "least",
    "only", "just", "barely", "hardly", "still",
    "without"
}

# Preserve these words during stopword removal
stop_words = stop_words - important_words



def clean_text(text):
    """
    Clean a review while preserving important sentiment words.
    """

    # Convert text to lowercase
    text = str(text).lower()

    # Expand common contractions before removing punctuation
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "can't": "cannot",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "won't": "will not",
        "mustn't": "must not"
    }

    for contraction, expanded_form in contractions.items():
        text = text.replace(contraction, expanded_form)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only English letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize the text
    words = text.split()

    # Remove stopwords while preserving negations
    words = [
        word for word in words
        if word not in stop_words and len(word) > 1
    ]
    # Join tokens back into a sentence
    return " ".join(words)


def main():

    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original records: {len(df)}")

    # Apply preprocessing to review text
    df["cleaned_content"] = df["content"].apply(clean_text)

    # Count reviews that became empty
    empty_reviews = (df["cleaned_content"].str.strip() == "").sum()

    print(f"Reviews empty after preprocessing: {empty_reviews}")

    # Remove reviews that contain no usable text
    df = df[df["cleaned_content"].str.strip() != ""].copy()

    # Save the preprocessed dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nPreprocessing completed successfully!")

    print(f"Final records: {len(df)}")
    print(f"Output saved at: {OUTPUT_PATH}")

    # Display sample before-and-after results
    print("\nSample preprocessing results:")

    for _, row in df.head(5).iterrows():
        print("\nOriginal:", row["content"])
        print("Processed:", row["cleaned_content"])

    # Verify the output
    print("\nVerification:")
    print("Missing values:")
    print(df.isnull().sum())

    print("\nDataset shape:", df.shape)


if __name__ == "__main__":
    main()