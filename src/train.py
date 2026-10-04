from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save charts without opening windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import get_data

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
COLUMNS_PATH = ROOT / "models" / "symptom_columns.pkl"
CHART_PATH = ROOT / "reports" / "model_comparison.png"

RANDOM_STATE = 42


def get_models():
    return {
        "Naive Bayes": BernoulliNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE), 
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def plot_comparison(results):
    names = list(results)
    cv_means = [results[n]["cv_mean"] * 100 for n in names]
    cv_stds = [results[n]["cv_std"] * 100 for n in names]
    test_accs = [results[n]["test_acc"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, cv_means, width, yerr=cv_stds, capsize=4,
            label="5-fold CV accuracy", color="steelblue")
    plt.bar(x + width / 2, test_accs, width,
            label="Test accuracy", color="darkorange")
    plt.xticks(x, names)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.title("Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    CHART_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()


def main():
    X, y, X_train, X_test, y_train, y_test = get_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    print(f"{'Model':<15} {'CV accuracy':<18} {'Test accuracy'}")
    print("-" * 50)

    for name, model in get_models().items():
        # 1. cross-validation on all data
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        # 2. fit on train split, 3. evaluate on test split
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results[name] = {
            "model": model,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "test_acc": test_acc,
        }
        print(f"{name:<15} {scores.mean():.3f} (+/- {scores.std():.3f})"
              f"    {test_acc:.3f}")

    # Pick the best by CV accuracy (>= so ties go to the later, more robust model)
    best_name = None
    best_cv = -1
    for name, r in results.items():
        if r["cv_mean"] >= best_cv:
            best_cv = r["cv_mean"]
            best_name = name

    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)
    plot_comparison(results)

    print(f"Saved model   -> {MODEL_PATH}")
    print(f"Saved columns -> {COLUMNS_PATH}")
    print(f"Saved chart   -> {CHART_PATH}")


if __name__ == "__main__":
    main()