"""Tests for the sklearn Pipeline: probabilities, determinism, save/load."""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import TARGET, clean
from src.features import FEATURES, build_preprocessor


def _fit(df):
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
    return pipe.fit(df[FEATURES], df[TARGET])


def test_pipeline_outputs_valid_probabilities(raw_df):
    df = clean(raw_df)
    proba = _fit(df).predict_proba(df[FEATURES])
    assert proba.shape == (len(df), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_is_deterministic(raw_df):
    df = clean(raw_df)
    a = _fit(df).predict_proba(df[FEATURES])
    b = _fit(df).predict_proba(df[FEATURES])
    assert (a == b).all()


def test_saved_model_roundtrip(raw_df, tmp_path):
    df = clean(raw_df)
    model = _fit(df)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    assert (reloaded.predict(df[FEATURES]) == model.predict(df[FEATURES])).all()
