from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
COLUMNS_PATH = ROOT / "models" / "symptom_columns.pkl"


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the model and column order once, then reuse."""
    return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)


def build_vector(selected_symptoms):
    """One-row DataFrame with the exact training column order."""
    _, columns = load_artifacts()
    chosen = set(selected_symptoms)
    return pd.DataFrame([[1 if c in chosen else 0 for c in columns]], columns=columns)


def predict_top_k(selected_symptoms, k=3):
    """Return the top-k (disease, confidence %) pairs, best first."""
    if not selected_symptoms:
        raise ValueError("Select at least one symptom.")
    model, _ = load_artifacts()
    probs = model.predict_proba(build_vector(selected_symptoms))[0]
    top_idx = probs.argsort()[::-1][:k]
    return [(model.classes_[i], round(float(probs[i]) * 100, 2)) for i in top_idx]


def predict_disease(selected_symptoms):
    """Best single prediction: (disease, confidence %)."""
    return predict_top_k(selected_symptoms, k=1)[0]


if __name__ == "__main__":
    from utils.precautions import get_precautions

    sample = ["itching", "skin_rash", "nodal_skin_eruptions"]
    print("Symptoms:", sample)
    for disease, conf in predict_top_k(sample):
        print(f"  {disease}: {conf}%")
    best = predict_disease(sample)[0]
    print("Precautions for", best, "->", get_precautions(best))