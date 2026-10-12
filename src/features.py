"""Feature definitions and the preprocessing transformer."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Continuous measurements -> impute median, scale
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# Already 0/1 -> impute most frequent, pass through unscaled
BINARY = ["sex", "fbs", "exang"]
# Codes with no natural order -> impute most frequent, one-hot
CATEGORICAL = ["cp", "restecg", "slope", "ca", "thal"]

FEATURES = NUMERIC + BINARY + CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # ignore unseen codes at serving time instead of crashing
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("bin", binary, BINARY),
        ("cat", categorical, CATEGORICAL),
    ])
