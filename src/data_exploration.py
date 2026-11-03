
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "raw" / "google_play_reviews.csv"
PLOTS_PATH = PROJECT_ROOT / "results" / "plots"

PLOTS_PATH.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load the Google Play reviews dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df


def explore_dataset(df):
    """Display basic information about the dataset."""

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    print("\n========== COLUMN INFORMATION ==========")
    df.info()

    print("\n========== FIRST FIVE REVIEWS ==========")
    print(df[["content", "score", "appId"]].head())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ANALYSIS ==========")
    print("Complete duplicate rows:", df.duplicated().sum())
    print("Duplicate review texts:", df["content"].duplicated().sum())
    print("Unique review texts:", df["content"].nunique())

    print("\n========== RATING DISTRIBUTION ==========")
    print(df["score"].value_counts().sort_index())

    print("\n========== APPLICATION DISTRIBUTION ==========")
    print(df["appId"].value_counts())

    # Create rating-derived sentiment labels for exploration
    sentiment_mapping = {
        1: "Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Positive"
    }

    df["sentiment"] = df["score"].map(sentiment_mapping)

    print("\n========== SENTIMENT DISTRIBUTION ==========")
    print(df["sentiment"].value_counts())

    
    # Investigate repeated reviews with conflicting ratings
    print("\n========== CONFLICTING DUPLICATE REVIEWS ==========")

    duplicate_analysis = (
        df.groupby("content")
        .agg(
            review_count=("score", "size"),
            unique_ratings=("score", "nunique"),
            ratings=("score", lambda x: sorted(x.unique()))
        )
        .reset_index()
    )

    conflicting_reviews = duplicate_analysis[
        duplicate_analysis["unique_ratings"] > 1
    ]

    print("Repeated texts with conflicting ratings:",
          len(conflicting_reviews))

    print("\nExamples:")
    print(conflicting_reviews.head(10).to_string(index=False))
    
    # Save conflicting reviews for manual inspection
    conflicting_reviews.to_csv(
    PROJECT_ROOT / "results" / "conflicting_reviews.csv",
    index=False
)

    print("\nConflicting reviews saved successfully!")  

    # Application-wise sentiment distribution
    print("\n========== APP-WISE SENTIMENT DISTRIBUTION ==========")

    app_sentiment = pd.crosstab(
        df["appId"],
        df["sentiment"]
    )

    print(app_sentiment.to_string())

    # Application-wise sentiment percentages
    print("\n========== APP-WISE SENTIMENT PERCENTAGES ==========")

    app_sentiment_percentage = pd.crosstab(
        df["appId"],
        df["sentiment"],
        normalize="index"
    ) * 100

    print(app_sentiment_percentage.round(2).to_string())

    # Review length analysis
    print("\n========== REVIEW LENGTH ==========")

    df["review_length"] = df["content"].str.split().str.len()
    print(df["review_length"].describe())

    print("\n========== REVIEW LENGTH ==========")
    df["review_length"] = df["content"].str.split().str.len()
    print(df["review_length"].describe())


def create_visualizations(df):
    """Generate exploratory data analysis plots."""

    # Rating distribution
    df["score"].value_counts().sort_index().plot(
        kind="bar",
        title="Distribution of Star Ratings",
        xlabel="Star Rating",
        ylabel="Number of Reviews"
    )

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "rating_distribution.png")
    plt.close()

    # Sentiment distribution
    sentiment_mapping = {
        1: "Negative",
        2: "Negative",
        3: "Neutral",
        4: "Positive",
        5: "Positive"
    }

    sentiment = df["score"].map(sentiment_mapping)

    sentiment.value_counts().reindex(
        ["Negative", "Neutral", "Positive"]
    ).plot(
        kind="bar",
        title="Sentiment Distribution",
        xlabel="Sentiment",
        ylabel="Number of Reviews"
    )

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "sentiment_distribution.png")
    plt.close()

    # Application-wise review distribution
    df["appId"].value_counts().sort_values().plot(
        kind="barh",
        title="Reviews per Application",
        xlabel="Number of Reviews",
        ylabel="Application"
    )

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "application_distribution.png")
    plt.close()

    print("\nVisualizations saved in results/plots/")


def main():
    df = load_dataset()
    explore_dataset(df)
    create_visualizations(df)


if __name__ == "__main__":
    main()
