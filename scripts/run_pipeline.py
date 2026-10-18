"""End-to-end ML pipeline runner.

Runs every stage in order with a clear header so you can see exactly
what is happening in the video demo.

Usage:
    python scripts/run_pipeline.py            # full pipeline
    python scripts/run_pipeline.py --skip-download  # if data already exists
"""
import argparse
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────── helpers ────────────────────────────────────────

def banner(step: int, title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  STEP {step}: {title}")
    print("=" * 60)


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    """Run a subprocess; exit the whole script if it fails."""
    print(f"\n▶  {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n✗  Command failed with exit code {result.returncode}. Aborting.", file=sys.stderr)
        sys.exit(result.returncode)


# ─────────────────────────── pipeline stages ────────────────────────────────

def step_download(python: str) -> None:
    banner(1, "Download raw dataset (UCI Cleveland Heart Disease)")
    run([python, "scripts/download_data.py"])


def step_clean(python: str) -> None:
    banner(2, "Clean & binarise target → data/processed/heart_clean.csv")
    run([python, "-m", "src.data"])


def step_eda(python: str) -> None:
    banner(3, "Exploratory Data Analysis → reports/figures/")
    run([python, "-m", "src.eda"])


def step_train(python: str) -> None:
    banner(4, "Train & tune models (LR, RF, GBM) — tracking in MLflow")
    print(textwrap.dedent("""\
        Models being compared:
          • Logistic Regression  (GridSearchCV over C, class_weight)
          • Random Forest        (GridSearchCV over n_estimators, max_depth, min_samples_leaf)
          • Gradient Boosting    (GridSearchCV over n_estimators, learning_rate, max_depth)

        All runs logged to:  mlflow.db  (SQLite)
        Best model saved to: models/model.joblib
        Metrics saved to:    models/metrics.json
    """))
    run([python, "-m", "src.train"])


def step_summary(python: str) -> None:
    banner(5, "Results summary — best model & metrics")
    run([python, "-c", textwrap.dedent("""\
        import json
        from pathlib import Path

        ROOT = Path('.')
        metrics_path = ROOT / 'models' / 'metrics.json'
        if not metrics_path.exists():
            print("models/metrics.json not found — did training complete?")
            raise SystemExit(1)

        m = json.loads(metrics_path.read_text())
        best = m['best_model']

        col_w = 24
        print()
        print(f"{'Model':<{col_w}} {'CV AUC':>8} {'Test AUC':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}")
        print("-" * (col_w + 46))
        for name, r in m['results'].items():
            marker = " ⭐" if name == best else ""
            print(
                f"{name:<{col_w}}"
                f"{r['cv_roc_auc']:>8.3f}"
                f"{r['roc_auc']:>10.3f}"
                f"{r['recall']:>8.3f}"
                f"{r['f1']:>8.3f}"
                f"{r['accuracy']:>10.3f}"
                f"{marker}"
            )
        print()
        print(f"Best model: {best}")
        print(f"Model file: models/model.joblib")
        print()
        print("To view all MLflow runs:")
        print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print("  → open http://127.0.0.1:5000")
    """)])


# ─────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full heart-disease ML pipeline.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the data download step (use if raw data already exists).",
    )
    args = parser.parse_args()

    # Use the same Python interpreter that launched this script
    python = sys.executable

    t0 = time.perf_counter()
    print("\n🚀  Heart Disease MLOps Pipeline — starting all stages\n")

    if not args.skip_download:
        step_download(python)
    else:
        print("\n⏭  Step 1 skipped (--skip-download flag set)\n")

    step_clean(python)
    step_eda(python)
    step_train(python)
    step_summary(python)

    elapsed = time.perf_counter() - t0
    print(f"\n✅  All pipeline stages completed in {elapsed:.1f}s")
    print("\nNext steps:")
    print("  mlflow ui --backend-store-uri sqlite:///mlflow.db   → experiment browser")
    print("  uvicorn api.app:app --reload --port 8000            → start API")
    print("  pytest -v                                           → run all tests")


if __name__ == "__main__":
    main()
