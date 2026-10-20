"""Train, tune and compare models; track everything in MLflow.

Usage: python -m src.train
Outputs:
  models/model.joblib    best full pipeline (preprocessing + classifier)
  models/metrics.json    test metrics of every candidate
  mlflow.db + mlruns/    MLflow tracking store (view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`)
"""
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.data import ROOT, TARGET, load_clean  # noqa: E402
from src.features import FEATURES, build_preprocessor  # noqa: E402

SEED = 42
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "reports" / "runs"

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 5, 8],
         "clf__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [2, 3]},
    ),
}


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_plots(model, X, y, name: str) -> list:
    out = ARTIFACT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    ConfusionMatrixDisplay.from_estimator(model, X, y, cmap="Blues")
    plt.title(f"{name} - confusion matrix")
    paths.append(out / "confusion_matrix.png")
    plt.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_estimator(model, X, y)
    plt.title(f"{name} - ROC curve")
    paths.append(out / "roc_curve.png")
    plt.savefig(paths[-1], dpi=110, bbox_inches="tight")
    plt.close()
    return paths


def main() -> dict:
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("heart-disease-classification")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, best_name, best_model, best_run_id = {}, None, None, None
    for name, (clf, grid) in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)

            test_metrics = evaluate(search.best_estimator_, X_test, y_test)
            mlflow.log_param("model_type", name)
            mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_roc_auc", search.best_score_)
            mlflow.log_metric("cv_roc_auc_std",
                              search.cv_results_["std_test_score"][search.best_index_])
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            for p in save_plots(search.best_estimator_, X_test, y_test, name):
                mlflow.log_artifact(str(p))
            mlflow.sklearn.log_model(
                search.best_estimator_, name="model", input_example=X_test.head(2),
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

            results[name] = {"cv_roc_auc": search.best_score_, **test_metrics,
                             "best_params": search.best_params_,
                             "run_id": run.info.run_id}
            mlflow.set_tag("best_model", "false")
            print(f"{name:20s} cv_auc={search.best_score_:.3f} "
                  f"test_auc={test_metrics['roc_auc']:.3f} recall={test_metrics['recall']:.3f}")
            # use CV score for selection to keep test set truly held-out
            if best_name is None or search.best_score_ > results[best_name]["cv_roc_auc"]:
                best_name, best_model = name, search.best_estimator_
                best_run_id = run.info.run_id

    # mark the best run in MLflow so it's easy to find
    client = mlflow.MlflowClient()
    client.set_tag(best_run_id, "best_model", "true")
    client.set_tag(best_run_id, "selected_reason", "highest_cv_roc_auc")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "model.joblib")
    summary = {"best_model": best_name, "results": results}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Best model: {best_name} -> {MODEL_DIR / 'model.joblib'}")
    return summary


if __name__ == "__main__":
    main()
