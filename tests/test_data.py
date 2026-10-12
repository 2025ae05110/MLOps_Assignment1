"""Tests for src/data.py and src/features.py."""
import numpy as np
import pandas as pd

from src.data import TARGET, clean
from src.features import FEATURES, build_preprocessor


def test_target_is_binary(raw_df):
    df = clean(raw_df)
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_target_binarisation(raw_df):
    """num > 0 must map to 1, num == 0 to 0."""
    df_raw = raw_df.copy()
    df = clean(df_raw)
    expected = (df_raw["num"] > 0).astype(int).values
    assert (df[TARGET].values == expected).all()


def test_all_features_numeric(raw_df):
    df = clean(raw_df)
    for col in FEATURES:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"


def test_preprocessor_removes_nan(raw_df):
    df = clean(raw_df)
    prep = build_preprocessor()
    out = prep.fit_transform(df[FEATURES])
    assert not np.isnan(out).any(), "NaN values remain after preprocessing"


def test_unseen_categories_handled(raw_df):
    """OneHotEncoder with handle_unknown='ignore' must not raise on new codes."""
    df = clean(raw_df)
    prep = build_preprocessor()
    prep.fit(df[FEATURES])
    # introduce an unseen code for a categorical column
    test_row = df[FEATURES].iloc[[0]].copy()
    test_row["cp"] = 99
    out = prep.transform(test_row)
    assert not np.isnan(out).any()
