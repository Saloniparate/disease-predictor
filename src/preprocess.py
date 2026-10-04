from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "dataset.csv"
CLEAN_PATH = ROOT / "data" / "cleaned_dataset.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as the EDA notebook."""
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("__", "_")
    )
    df = df.drop(columns=[c for c in df.columns if c.endswith(".1")])
    df["prognosis"] = df["prognosis"].str.strip()
    df = df.fillna(0).drop_duplicates().reset_index(drop=True)
    return df


def load_data(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned CSV, or rebuild it from the raw file if missing."""
    if Path(path).exists():
        return pd.read_csv(path)
    df = clean_data(pd.read_csv(RAW_PATH))
    df.to_csv(CLEAN_PATH, index=False)
    return df


def preprocess(df: pd.DataFrame):
    """Split the dataframe into features (X) and target (y)."""
    X = df.drop(columns="prognosis")
    y = df["prognosis"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split so every disease appears in train and test."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_data():
    """Convenience function used by train.py and evaluate.py."""
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return X, y, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y, X_train, X_test, y_train, y_test = get_data()
    print("Features:", X.shape)
    print("Diseases:", y.nunique())
    print("Train:", X_train.shape, "| Test:", X_test.shape)