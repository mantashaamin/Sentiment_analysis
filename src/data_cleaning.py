
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "dataset" / "raw" / "google_play_reviews.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "processed" / "cleaned_reviews.csv"
RESULTS_PATH = PROJECT_ROOT / "results"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def clean_dataset(df):

    print("\n========== PHASE 2: DATA CLEANING ==========")

    original_count = len(df)

    # Keep only columns required for sentiment analysis
    df = df[["content", "score", "appId"]].copy()

    print("\nOriginal records:", original_count)

    # Remove missing and empty review texts
    df["content"] = df["content"].astype("string").str.strip()

    df = df.dropna(subset=["content", "score", "appId"])
    df = df[df["content"] != ""]

    print("Records after removing missing/empty values:", len(df))

    # Find conflicting review texts
    rating_counts = df.groupby("content")["score"].nunique()

    conflicting_texts = rating_counts[
        rating_counts > 1
    ].index

    conflicting_records = df[
        df["content"].isin(conflicting_texts)
    ].copy()

    print("\nConflicting review groups:", len(conflicting_texts))
    print("Conflicting records:", len(conflicting_records))

    # Save conflicting records separately
    conflicting_records.to_csv(
        RESULTS_PATH / "conflicting_review_records.csv",
        index=False
    )

    # Exclude all records belonging to conflicting text groups
    df = df[
        ~df["content"].isin(conflicting_texts)
    ].copy()

    print("Records after conflict removal:", len(df))

    # Remove exact duplicate rows
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["content", "score", "appId"]
    )

    removed_duplicates = before_duplicates - len(df)

    print("Exact duplicate records removed:", removed_duplicates)

    # Convert star ratings into sentiment labels
    def assign_sentiment(score):

        if score <= 2:
            return "Negative"
        elif score == 3:
            return "Neutral"
        else:
            return "Positive"

    df["sentiment"] = df["score"].apply(assign_sentiment)

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== CLEANING SUMMARY ==========")
    print("Final records:", len(df))
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\nCleaned dataset saved to:")
    print(OUTPUT_PATH)

    return df



def main():

    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    clean_dataset(df)

    # Verify the saved cleaned dataset
    cleaned_df = pd.read_csv(OUTPUT_PATH)

    print("\n========== FINAL DATASET VERIFICATION ==========")

    print("Dataset shape:", cleaned_df.shape)

    print("\nMissing values:")
    print(cleaned_df.isnull().sum())

    print("\nDuplicate combinations:")
    print(
        cleaned_df.duplicated(
            subset=["content", "score", "appId"]
        ).sum()
    )

    print("\nSentiment distribution:")
    print(cleaned_df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
