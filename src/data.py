"""Load and clean the raw UCI Cleveland heart disease data."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_clean.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """The raw file has no header and uses '?' for missing values."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarise the target and fix dtypes.

    Missing values (only in `ca` and `thal`, ~6 rows) are left as NaN on
    purpose: imputation happens inside the model pipeline so the exact same
    logic runs at training and serving time.
    """
    df = df.copy().drop_duplicates()
    # num: 0 = no disease, 1-4 = increasing severity -> collapse to binary
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_clean(raw_path: Path = RAW_PATH, save: bool = True) -> pd.DataFrame:
    df = clean(load_raw(raw_path))
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
    return df


if __name__ == "__main__":
    data = load_clean()
    print(data.shape)
    print(data.isna().sum()[data.isna().sum() > 0])
    print(data[TARGET].value_counts())
