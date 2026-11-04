
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_DIR = PROJECT_ROOT / "dataset" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


y_train = pd.read_csv(FEATURES_DIR / "y_train.csv")["sentiment"]
y_test = pd.read_csv(FEATURES_DIR / "y_test.csv")["sentiment"]


def get_models():

    return {
        "Naive Bayes": GaussianNB(),

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


def main():

    all_results = []

    for feature_type in ["cbow", "skipgram"]:

        print("\n================================")
        print(f"WORD2VEC: {feature_type.upper()}")
        print("================================")

        X_train = np.load(
            FEATURES_DIR / f"X_train_{feature_type}.npy"
        )

        X_test = np.load(
            FEATURES_DIR / f"X_test_{feature_type}.npy"
        )

        print("Training shape:", X_train.shape)
        print("Testing shape:", X_test.shape)

        models = get_models()

        for name, model in models.items():

            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            filename = (
                f"{name.lower().replace(' ', '_')}_{feature_type}.pkl"
            )

            joblib.dump(model, MODELS_DIR / filename)

            all_results.append({
                "Model": name,
                "Feature": feature_type.upper(),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    results_df.to_csv(
        RESULTS_DIR / "word2vec_comparison.csv",
        index=False
    )

    print("\n========== WORD2VEC COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\nWord2Vec training completed!")


if __name__ == "__main__":
    main()
