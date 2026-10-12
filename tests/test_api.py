"""Tests for the FastAPI endpoints: /health, /predict, /metrics."""
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import api.app as app_module
from src.data import TARGET, clean
from src.features import FEATURES, build_preprocessor


def _make_model(df_clean):
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000))])
    return pipe.fit(df_clean[FEATURES], df_clean[TARGET])


@pytest.fixture(scope="session")
def trained_model():
    """Train once per session; build synthetic data inline (no cross-scope dependency)."""
    import numpy as np
    import pandas as pd
    from src.data import COLUMNS
    rng = np.random.default_rng(0)
    n = 120
    df_raw = pd.DataFrame({
        "age": rng.integers(29, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n), "num": rng.integers(0, 5, n),
    })[COLUMNS]
    df_raw.loc[[3, 10], "ca"] = np.nan
    df_raw.loc[[5], "thal"] = np.nan
    return _make_model(clean(df_raw))


@pytest.fixture
def client(trained_model, monkeypatch):
    """TestClient with the trained model injected directly into the app module.

    We never reload the module (which would re-register Prometheus metrics and
    raise DuplicateTimeseries). Instead we monkey-patch the module-level
    ``model`` variable directly and restore it afterwards.
    """
    monkeypatch.setattr(app_module, "model", trained_model)
    monkeypatch.setattr(app_module, "MODEL_VERSION", "test-1.0")
    return TestClient(app_module.app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_response_shape(client, sample_patient):
    resp = client.post("/predict", json=sample_patient)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("heart disease risk", "no heart disease")
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["probability_disease"] <= 1.0
    assert "model_version" in data


def test_predict_confidence_geq_half(client, sample_patient):
    """Confidence must always be ≥ 0.5 (it's the max-class probability)."""
    resp = client.post("/predict", json=sample_patient)
    assert resp.status_code == 200
    assert resp.json()["confidence"] >= 0.5


def test_predict_missing_field(client, sample_patient):
    payload = {k: v for k, v in sample_patient.items() if k != "age"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_age(client, sample_patient):
    payload = {**sample_patient, "age": 999}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_sex(client, sample_patient):
    payload = {**sample_patient, "sex": 5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_metrics_endpoint_exposes_counter(client, sample_patient):
    """After a prediction, /metrics must contain the custom counter."""
    client.post("/predict", json=sample_patient)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "heart_predictions_total" in resp.text
