"""Feature definitions and the preprocessing transformer."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# continuous features: median impute + scale
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# binary 0/1 flags: most-frequent impute, no scaling needed
BINARY = ["sex", "fbs", "exang"]
# nominal codes: most-frequent impute + one-hot encode
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
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("bin", binary, BINARY),
        ("cat", categorical, CATEGORICAL),
    ])
