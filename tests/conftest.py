import numpy as np
import pandas as pd
import pytest

from src.data import COLUMNS


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small synthetic frame with the raw UCI schema (incl. NaNs and num 0-4)."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.integers(29, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n), "num": rng.integers(0, 5, n),
    })[COLUMNS]
    df.loc[[3, 10], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df


@pytest.fixture
def sample_patient() -> dict:
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}
