
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_DIR = PROJECT_ROOT / "dataset" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading test data...")

    X_test = load_npz(
        FEATURES_DIR / "X_test_tfidf.npz"
    )

    y_test = pd.read_csv(
        FEATURES_DIR / "y_test.csv"
    )["sentiment"]

    print("Test matrix shape:", X_test.shape)

    # Load the best model
    model_path = MODELS_DIR / "logistic_regression_tfidf.pkl"

    model = joblib.load(model_path)

    print("\nBest model loaded: Logistic Regression + TF-IDF")

    # Generate predictions
    predictions = model.predict(X_test)

    # Classification report
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Confusion matrix
    labels = ["Negative", "Neutral", "Positive"]

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Logistic Regression + TF-IDF")
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nEvaluation completed successfully!")

    print("Classification report saved to results/")
    print("Confusion matrix saved to results/plots/")


if __name__ == "__main__":
    main()
