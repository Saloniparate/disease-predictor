from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocess import get_data

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
REPORT_PATH = ROOT / "reports" / "classification_report.txt"
CM_PATH = ROOT / "reports" / "confusion_matrix.png"


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(20, 17))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 6},
        cbar=False,
    )
    plt.title("Confusion Matrix (Test Set)", fontsize=16)
    plt.xlabel("Predicted disease")
    plt.ylabel("Actual disease")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()


def main():
    _, _, _, X_test, _, y_test = get_data()
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    print(f"Model: {type(model).__name__}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    REPORT_PATH.write_text(report)
    print("\n" + report)

    plot_confusion_matrix(y_test, y_pred, list(model.classes_))
    print(f"Saved report -> {REPORT_PATH}")
    print(f"Saved matrix -> {CM_PATH}")


if __name__ == "__main__":
    main()