# Heart Disease Risk Prediction — End-to-End MLOps Pipeline

**Course:** Machine Learning Operations (MLOps) – AIMLCZG523, BITS Pilani WILP  
**Assignment:** 01 – End-to-End ML Model Development, CI/CD and Production Deployment  
**Repository:** `https://github.com/2025ae05110/MLOps_Assignment1`

A binary classifier that predicts the risk of heart disease from patient health
data (UCI Cleveland Heart Disease dataset), packaged and deployed as a
containerised, monitored REST API on Kubernetes, with experiment tracking and a
CI/CD pipeline.

---

## Table of contents

1. [Tech stack](#2-tech-stack)
2. [Architecture](#3-architecture)
3. [Repository structure](#4-repository-structure)
4. [Dataset](#5-dataset)
5. [Step-by-step: run the whole project](#7-step-by-step-run-the-whole-project)
6. [API reference](#9-api-reference)
7. [CI/CD pipeline](#10-cicd-pipeline)
8. [Kubernetes deployment](#11-kubernetes-deployment)
9. [Monitoring and logging](#12-monitoring-and-logging)
10. [Testing](#13-testing)
11. [Results](#18-results)

---

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| Data / ML | pandas, numpy, scikit-learn |
| Visualisation | matplotlib, seaborn |
| Data source | `ucimlrepo` (UCI dataset id 45) |
| Experiment tracking | MLflow (SQLite backend) |
| Model format | joblib (sklearn Pipeline) + MLflow model |
| API | FastAPI + Uvicorn, Pydantic v2 validation |
| Testing / lint | pytest, httpx (TestClient), ruff |
| CI/CD | GitHub Actions |
| Container | Docker (python:3.11-slim, non-root) |
| Orchestration | Kubernetes (Minikube / Docker Desktop) |
| Monitoring | Prometheus + Grafana, `prometheus-fastapi-instrumentator`, JSON request logs |

---

## Architecture

```
                ┌───────────────────────── Development ─────────────────────────┐
 UCI repo ──► scripts/download_data.py ──► src/data.py ──► src/eda.py (figures)
                                               │
                                               ▼
                                     src/features.py (ColumnTransformer)
                                               │
                                               ▼
                         src/train.py  ── GridSearchCV × 3 models ──► MLflow (mlflow.db)
                                               │                      params, metrics,
                                               ▼                      plots, models
                                     models/model.joblib  (best full pipeline)
                └───────────────────────────────────────────────────────────────┘

                ┌──────────────────────── CI/CD (GitHub Actions) ────────────────┐
 git push ──► lint (ruff) ──► pytest ──► download + train ──► quality gate (AUC ≥ 0.80)
                                                   ──► docker build ──► container smoke test
                                                   ──► artifacts (model, reports, logs)
                └───────────────────────────────────────────────────────────────┘

                ┌──────────────────────────── Serving ───────────────────────────┐
 Client ──► LoadBalancer Service ──► Deployment (2 pods, FastAPI)
                                       /predict  /health  /metrics
                                               │
                                     Prometheus (scrape /metrics) ──► Grafana dashboard
                                     stdout JSON logs ──► kubectl logs / docker logs
                └───────────────────────────────────────────────────────────────┘
```

---

## Repository structure

```
heart-disease-mlops/
├── README.md
├── requirements.txt             # full dev/training/CI dependencies
├── requirements-serve.txt       # minimal dependencies for the Docker image
├── Dockerfile                   # serving image (FastAPI)
├── docker-compose.yml           # local stack: api + prometheus + grafana
├── sample_request.json          # example /predict payload
├── pytest.ini                   # pytest config (pythonpath = .)
├── ruff.toml                    # lint config
├── .gitignore / .dockerignore
│
├── scripts/
│   └── download_data.py         # fetch dataset (ucimlrepo → fallback URLs)
│
├── src/
│   ├── __init__.py
│   ├── data.py                  # load raw file, clean, binarise target, save CSV
│   ├── features.py              # feature lists + preprocessing ColumnTransformer
│   ├── eda.py                   # EDA figures → reports/figures/
│   └── train.py                 # tune 3 models, log to MLflow, save best model
│
├── api/
│   ├── __init__.py
│   └── app.py                   # FastAPI app: /predict, /health, /metrics, logging
│
├── tests/
│   ├── conftest.py              # synthetic raw data + sample patient fixtures
│   ├── test_data.py             # cleaning + preprocessing tests
│   ├── test_model.py            # pipeline, determinism, save/load tests
│   └── test_api.py              # endpoint, validation and metrics tests
│
├── .github/workflows/
│   └── ci.yml                   # lint → test → train → quality gate → docker
│
├── k8s/
│   ├── deployment.yaml          # 2 replicas, probes, resource limits
│   └── service.yaml             # LoadBalancer on port 80 → 8000
│
├── monitoring/
│   ├── prometheus.yml           # scrape config for the API
│   └── grafana-datasource.yml   # auto-provisioned Prometheus datasource
│
├── data/
│   ├── raw/                     # downloaded file (git-ignored)
│   └── processed/heart_clean.csv
├── models/                      # model.joblib, metrics.json (generated)
├── reports/figures/             # EDA figures (generated)
└── screenshots/                 # evidence for the report
```

---

## Dataset

- **Source:** UCI Machine Learning Repository – Heart Disease (Cleveland), id 45
- **Rows:** 303 patients, **14 columns** (13 features + `num`)
- **Missing values:** `ca` (4 rows) and `thal` (2 rows), marked `?` in the raw file
- **Target:** `num` ∈ {0,1,2,3,4} → binarised to `target = 1 if num > 0 else 0`

---

## Step-by-step: run the whole project

```bash
# Step 0 — Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw data/processed models reports/figures screenshots

# Step 1 — Get the data
python scripts/download_data.py

# Step 2 — Clean the data
python -m src.data

# Step 3 — EDA
python -m src.eda

# Step 4 — Train and track experiments
python -m src.train
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000

# Step 5 — Lint and test
ruff check .
pytest -v

# Step 6 — Run API locally
uvicorn api.app:app --reload --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json

# Step 7 — Docker
docker build -t heart-api:1.0 .
docker run -d -p 8000:8000 --name heart-api heart-api:1.0
curl localhost:8000/health

# Step 8 — Kubernetes (Minikube)
minikube start --driver=docker
minikube image load heart-api:1.0
kubectl apply -f k8s/
kubectl get pods,svc
minikube tunnel   # keep open in a second terminal

# Step 9 — Monitoring stack
docker compose up --build -d
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_version": "..."}` |
| POST | `/predict` | Predict heart disease risk for one patient |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

**Request** (`POST /predict`):
```json
{"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
 "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
 "ca": 0, "thal": 6}
```

**Response**:
```json
{"prediction": 1, "label": "heart disease risk", "confidence": 0.84,
 "probability_disease": 0.84, "model_version": "1.0"}
```

---

## CI/CD pipeline

File: `.github/workflows/ci.yml`. Triggers: push to `main`, pull requests, manual dispatch.

| Job | Steps | Artifacts |
|---|---|---|
| `lint-test` | checkout → Python 3.11 → install → `ruff check .` → `pytest` | `test-results` (JUnit XML) |
| `train` (needs lint-test) | download data → EDA → train → **quality gate: fail if test ROC-AUC < 0.80** | `model-and-reports` |
| `docker` (needs train) | download model artifact → `docker build` → run container → curl `/health` and `/predict` | `smoke-test` |

---

## Kubernetes deployment

```bash
kubectl get pods,svc
kubectl describe pod <pod-name>
kubectl logs -l app=heart-api -f
kubectl scale deployment heart-api --replicas=3
kubectl rollout restart deployment heart-api
kubectl delete -f k8s/
```

---

## Monitoring and logging

Grafana panel queries:
| Panel | PromQL |
|---|---|
| Request rate | `sum(rate(http_requests_total[1m])) by (handler)` |
| p95 latency | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))` |
| Predictions by class | `sum(heart_predictions_total) by (prediction)` |
| Error rate | `sum(rate(http_requests_total{status=~"4..\|5.."}[1m]))` |

---

## Testing

| File | What it checks |
|---|---|
| `test_data.py` | target is binary and correctly mapped, all features numeric, preprocessor removes NaN, unseen categories handled |
| `test_model.py` | valid probabilities, deterministic training, joblib save/load round-trip |
| `test_api.py` | `/health`, `/predict` response shape and confidence, 422 on invalid/missing input, `/metrics` exposes custom counter |

Run: `pytest -v`

---

## Results

| Model | CV ROC-AUC | Test Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.904** | 0.885 | 0.839 | **0.929** | 0.881 | **0.968** |
| Random Forest | 0.898 | 0.885 | 0.839 | 0.929 | 0.881 | 0.949 |
| Gradient Boosting | 0.875 | 0.885 | 0.839 | 0.929 | 0.881 | 0.950 |

**Selected model:** Logistic Regression — highest cross-validated ROC-AUC (0.904).  
Model selection is based on CV score only; the test set is reserved for final reporting.  
Recall is prioritised because a missed heart-disease case (false negative) is clinically costlier than a false alarm.
