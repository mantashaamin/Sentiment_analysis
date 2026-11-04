import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.sparse import save_npz
from gensim.models import Word2Vec
import numpy as np
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "preprocessed_reviews.csv"
)
FEATURES_DIR = PROJECT_ROOT / "dataset" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"


def tokenize_reviews(reviews):
    """
    Convert each review into a list of words.
    """
    return [review.split() for review in reviews]


def get_review_vectors(model, reviews, vector_size=100):
    """
    Generate one vector for each review by averaging
    the vectors of its known words.
    """

    review_vectors = []

    for review in reviews:

        words = review.split()

        word_vectors = [
            model.wv[word]
            for word in words
            if word in model.wv
        ]

        if word_vectors:
            review_vector = np.mean(word_vectors, axis=0)
        else:
            review_vector = np.zeros(vector_size)

        review_vectors.append(review_vector)

    return np.array(review_vectors)

def main():

    print("Loading preprocessed dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # Features and target
    X = df["cleaned_content"]
    y = df["sentiment"]

    # Group identical original reviews together
    groups = df["content"].str.lower().str.strip()

    # Create an 80:20 group-aware split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    print("\nTraining and testing split completed!")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining sentiment distribution:")
    print(y_train.value_counts())

    print("\nTesting sentiment distribution:")
    print(y_test.value_counts())

    # Verify no original review group overlaps
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    print("\nOverlapping review groups:", len(train_groups & test_groups))
    
        # -----------------------------------------
    # BAG OF WORDS FEATURE EXTRACTION
    # -----------------------------------------

    print("\n========== BAG OF WORDS ==========")

    # Initialize CountVectorizer
    bow_vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    # Learn vocabulary from training data
    X_train_bow = bow_vectorizer.fit_transform(X_train)

    # Transform testing data using training vocabulary
    X_test_bow = bow_vectorizer.transform(X_test)

    # Display feature matrix information
    print("\nBag-of-Words completed successfully!")

    print("Training feature matrix:", X_train_bow.shape)
    print("Testing feature matrix:", X_test_bow.shape)

    print("\nTotal vocabulary features:",
          len(bow_vectorizer.vocabulary_))

    # Display sample vocabulary
    print("\nSample vocabulary:")
    print(list(bow_vectorizer.vocabulary_.keys())[:20])

    # Display sample feature matrix
    print("\nSample feature matrix:")
    print(X_train_bow[:5].toarray())
    
        # -----------------------------------------
    # TF-IDF FEATURE EXTRACTION
    # -----------------------------------------

    print("\n========== TF-IDF ==========")

    # Initialize TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )

    # Learn vocabulary and calculate IDF using training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    # Transform testing data using training vocabulary and IDF
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print("\nTF-IDF completed successfully!")

    print("Training feature matrix:", X_train_tfidf.shape)
    print("Testing feature matrix:", X_test_tfidf.shape)

    print(
        "\nTotal vocabulary features:",
        len(tfidf_vectorizer.vocabulary_)
    )

    # Display sample vocabulary
    print("\nSample vocabulary:")
    print(list(tfidf_vectorizer.vocabulary_.keys())[:20])

    # Display sample TF-IDF values
    print("\nSample TF-IDF matrix:")
    print(X_train_tfidf[:5].toarray())
    
        # -----------------------------------------
    # SAVE FEATURE MATRICES AND VECTORIZERS
    # -----------------------------------------

    print("\n========== SAVING FEATURES ==========")

    # Create directories
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save BoW matrices
    save_npz(FEATURES_DIR / "X_train_bow.npz", X_train_bow)
    save_npz(FEATURES_DIR / "X_test_bow.npz", X_test_bow)

    # Save TF-IDF matrices
    save_npz(FEATURES_DIR / "X_train_tfidf.npz", X_train_tfidf)
    save_npz(FEATURES_DIR / "X_test_tfidf.npz", X_test_tfidf)

    # Save target labels
    y_train.to_csv(FEATURES_DIR / "y_train.csv", index=False)
    y_test.to_csv(FEATURES_DIR / "y_test.csv", index=False)

    # Save vectorizers
    joblib.dump(
        bow_vectorizer,
        MODELS_DIR / "bow_vectorizer.pkl"
    )

    joblib.dump(
        tfidf_vectorizer,
        MODELS_DIR / "tfidf_vectorizer.pkl"
    )

    print("\nAll feature files saved successfully!")

    print("Feature directory:", FEATURES_DIR)
    print("Model directory:", MODELS_DIR)
    
        # -----------------------------------------
    # WORD2VEC - CBOW
    # -----------------------------------------

    print("\n========== WORD2VEC CBOW ==========")

    # Tokenize training and testing reviews
    train_tokens = tokenize_reviews(X_train)
    test_tokens = tokenize_reviews(X_test)

    # Initialize CBOW model
    cbow_model = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        sg=0,
        seed=42,
        epochs=20
    )

    # Generate review-level vectors
    X_train_cbow = get_review_vectors(
        cbow_model, X_train
    )

    X_test_cbow = get_review_vectors(
        cbow_model, X_test
    )

    print("CBOW training matrix:", X_train_cbow.shape)
    print("CBOW testing matrix:", X_test_cbow.shape)
    print("CBOW vocabulary size:", len(cbow_model.wv))
    
        # -----------------------------------------
    # WORD2VEC - SKIP-GRAM
    # -----------------------------------------

    print("\n========== WORD2VEC SKIP-GRAM ==========")

    # Initialize Skip-gram model
    skipgram_model = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        sg=1,
        seed=42,
        epochs=20
    )

    # Generate review-level vectors
    X_train_skipgram = get_review_vectors(
        skipgram_model, X_train
    )

    X_test_skipgram = get_review_vectors(
        skipgram_model, X_test
    )

    print("Skip-gram training matrix:", X_train_skipgram.shape)
    print("Skip-gram testing matrix:", X_test_skipgram.shape)
    print("Skip-gram vocabulary size:", len(skipgram_model.wv))
    
        # Save Word2Vec feature matrices
    np.save(FEATURES_DIR / "X_train_cbow.npy", X_train_cbow)
    np.save(FEATURES_DIR / "X_test_cbow.npy", X_test_cbow)

    np.save(FEATURES_DIR / "X_train_skipgram.npy", X_train_skipgram)
    np.save(FEATURES_DIR / "X_test_skipgram.npy", X_test_skipgram)

    # Save Word2Vec models
    cbow_model.save(str(MODELS_DIR / "word2vec_cbow.model"))
    skipgram_model.save(str(MODELS_DIR / "word2vec_skipgram.model"))

    print("\nWord2Vec models and feature matrices saved successfully!")

if __name__ == "__main__":
    main()