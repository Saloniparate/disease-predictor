from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "symptom_precaution.csv"


def _norm(name: str) -> str:
    """Lowercase and collapse spaces so names match across both files."""
    return " ".join(str(name).lower().split())


_df = pd.read_csv(CSV_PATH)
_disease_col = _df.columns[0]

_lookup = {}
for _, row in _df.iterrows():
    items = [str(p).strip().capitalize() for p in row.iloc[1:] if pd.notna(p) and str(p).strip()]
    _lookup[_norm(row[_disease_col])] = items


def get_precautions(disease: str) -> list:
    """Return the precautions for a disease, or [] if none are found."""
    return _lookup.get(_norm(disease), [])