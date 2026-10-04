from pathlib import Path

import joblib

COLUMNS_PATH = Path(__file__).resolve().parent.parent / "models" / "symptom_columns.pkl"

# Exact column order the model was trained on
symptoms = joblib.load(COLUMNS_PATH)


def pretty(symptom: str) -> str:
    """'skin_rash' -> 'Skin rash' (for display in the UI)."""
    return symptom.replace("_", " ").capitalize()


# label shown in UI -> real column name
symptom_labels = {pretty(s): s for s in symptoms}