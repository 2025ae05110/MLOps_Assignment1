"""Download the Cleveland Heart Disease dataset from the UCI repository.

Usage:  python scripts/download_data.py
Output: data/raw/processed.cleveland.data

Primary source is the `ucimlrepo` package; raw-file URLs are fallbacks.
Also fixes the common macOS SSL issue
(CERTIFICATE_VERIFY_FAILED) by using certifi's CA bundle when available.
"""
import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

FILE_NAME = "processed.cleveland.data"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / FILE_NAME

DIRECT_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
              f"heart-disease/{FILE_NAME}")
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=30) as resp:
        return resp.read()


def from_direct() -> bytes:
    return _fetch(DIRECT_URL)


def from_zip() -> bytes:
    with zipfile.ZipFile(io.BytesIO(_fetch(ZIP_URL))) as zf:
        return zf.read(FILE_NAME)


def from_ucimlrepo() -> bytes:
    """Fetch via the official `ucimlrepo` package (dataset id 45).

    features = 13 clinical columns, targets = `num` (0-4). We write them in the
    same headerless, '?'-for-missing format as the raw UCI file so that
    src/data.py works identically whichever source was used.
    """
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
    ds = fetch_ucirepo(id=45)
    df = ds.data.features.join(ds.data.targets)
    expected = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df = df[expected]  # enforce column order expected by src/data.py
    return df.to_csv(header=False, index=False, na_rep="?").encode()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # ucimlrepo first (official UCI Python API), raw-file URLs as fallbacks
    for name, source in [("ucimlrepo package", from_ucimlrepo),
                         ("direct file", from_direct), ("zip archive", from_zip)]:
        try:
            print(f"Trying {name} ...")
            data = source()
            if len(data) < 1000:
                raise ValueError(f"response too small ({len(data)} bytes)")
            OUT.write_bytes(data)
            rows = data.count(b"\n")
            print(f"Saved {rows} rows to {OUT}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("\nAll sources failed. Manual fallback:\n"
          "  1. Open https://archive.ics.uci.edu/dataset/45/heart+disease in a browser\n"
          "  2. Click Download, unzip, and copy processed.cleveland.data to\n"
          f"     {OUT}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
