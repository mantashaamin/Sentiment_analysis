
from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_DIR = PROJECT_ROOT / "dataset" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load training and testing labels
y_train = pd.read_csv(FEATURES_DIR / "y_train.csv")["sentiment"]
y_test = pd.read_csv(FEATURES_DIR / "y_test.csv")["sentiment"]


def load_features(feature_type):

    X_train = load_npz(
        FEATURES_DIR / f"X_train_{feature_type}.npz"
    )

    X_test = load_npz(
        FEATURES_DIR / f"X_test_{feature_type}.npz"
    )

    return X_train, X_test


def get_models():

    models = {
        "Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    }

    return models


def main():

    all_results = []

    for feature_type in ["bow", "tfidf"]:

        print("\n====================================")
        print(f"FEATURE TYPE: {feature_type.upper()}")
        print("====================================")

        X_train, X_test = load_features(feature_type)

        print("Training matrix:", X_train.shape)
        print("Testing matrix:", X_test.shape)

        models = get_models()

        for model_name, model in models.items():

            print(f"\nTraining {model_name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            predictions = model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            print(f"\n{model_name} Results")
            print("------------------------------")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    predictions,
                    zero_division=0
                )
            )

            # Save trained model
            model_filename = (
                f"{model_name.lower().replace(' ', '_')}_{feature_type}.pkl"
            )

            joblib.dump(
                model,
                MODELS_DIR / model_filename
            )

            # Store results
            all_results.append({
                "Model": model_name,
                "Feature": feature_type.upper(),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            print(f"Model saved: {model_filename}")

    # Create comparison table
    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    results_df.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    print("\n====================================")
    print("FINAL MODEL COMPARISON")
    print("====================================")

    print(results_df.to_string(index=False))

    print("\nResults saved to results/model_comparison.csv")
    print("Model training completed successfully!")


if __name__ == "__main__":
    main()
