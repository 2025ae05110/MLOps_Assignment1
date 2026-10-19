"""FastAPI serving app for the heart disease model.

Run locally: uvicorn api.app:app --reload --port 8000
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.features import FEATURES

_default_model = str(Path(__file__).resolve().parents[1] / "models" / "model.joblib")
MODEL_PATH = Path(os.getenv("MODEL_PATH", _default_model))
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("heart-api")

PREDICTIONS = Counter("heart_predictions_total", "Predictions by class", ["prediction"])
CONFIDENCE = Histogram("heart_prediction_confidence", "Model confidence",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


class Patient(BaseModel):
    """Input schema; ranges reject obviously invalid clinical values."""
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="chest pain type")
    trestbps: float = Field(..., ge=50, le=250, description="resting BP (mm Hg)")
    chol: float = Field(..., ge=100, le=700, description="serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=50, le=250, description="max heart rate")
    exang: int = Field(..., ge=0, le=1, description="exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: int = Field(..., ge=0, le=3, description="major vessels coloured")
    thal: int = Field(..., description="3 = normal, 6 = fixed, 7 = reversible")

    model_config = {"json_schema_extra": {"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}}}


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    model_version: str


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        log.info(json.dumps({"event": "model_loaded", "path": str(MODEL_PATH)}))
    else:
        log.warning(json.dumps({"event": "model_missing", "path": str(MODEL_PATH)}))
    yield
    model = None


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One structured JSON log line per request."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    log.info(json.dumps({
        "event": "request", "id": request_id, "method": request.method,
        "path": request.url.path, "status": response.status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }))
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_not_loaded",
            "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    row = pd.DataFrame([patient.model_dump()])[FEATURES]
    proba = float(model.predict_proba(row)[0, 1])
    pred = int(proba >= 0.5)
    confidence = proba if pred == 1 else 1 - proba

    PREDICTIONS.labels(prediction=str(pred)).inc()
    CONFIDENCE.observe(confidence)
    log.info(json.dumps({"event": "prediction", "input": patient.model_dump(),
                         "prediction": pred, "confidence": round(confidence, 4)}))
    return Prediction(prediction=pred,
                      label="heart disease risk" if pred else "no heart disease",
                      confidence=round(confidence, 4),
                      probability_disease=round(proba, 4),
                      model_version=MODEL_VERSION)
