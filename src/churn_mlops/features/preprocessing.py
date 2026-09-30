import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_mlops.config import ID_COLUMN, TARGET

# Numeric column that is actually categorical in nature (only 0/1 values)
CATEGORICAL_NUMERIC_COLUMNS: list[str] = ["SeniorCitizen"]


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data — deterministic transforms that require no fitting.

    - Drop the ID column (carries no predictive signal)
    - Cast TotalCharges to numeric (the original dataset has a few blank strings)
    - Cast SeniorCitizen to category so OneHotEncoder handles it correctly
    """
    X = X.copy()

    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    for col in CATEGORICAL_NUMERIC_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype("object")

    return X


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Auto-detect categorical/numerical columns after clean_features has run."""
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    return categorical_features, numerical_features


def build_preprocessor(
    categorical_features: list[str],
    numerical_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer: impute+scale for numeric, one-hot for categorical.

    Returns an unfitted object — it gets fit inside the Pipeline at
    training time (see models/train.py) and applies the exact same
    transform at serving time, since it's part of the same Pipeline
    object that gets logged to MLflow.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def encode_target(y: pd.Series) -> pd.Series:
    """Map the 'Yes'/'No' label to 1/0."""
    return y.map({"No": 0, "Yes": 1}).astype(int)


def transform_to_dataframe(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series | None = None,
    target_name: str = TARGET,
) -> pd.DataFrame:
    """Apply a FITTED preprocessor and return a readable DataFrame.

    Used to persist the processed train/test sets under data/processed/
    (e.g. for reproducibility, debugging, or feeding a reference dataset
    to Evidently) — the raw ColumnTransformer.transform() output is a
    bare numpy array, so this re-attaches the generated feature names
    (from OneHotEncoder, etc.) and appends the target column back.

    Args:
        preprocessor: a ColumnTransformer that has already been fit
            (e.g. pipeline.named_steps["preprocessor"] after training).
        X: cleaned feature dataframe (output of clean_features).
        y: optional encoded target series to attach as a column.
        target_name: column name to use for the target.
    """
    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    if y is not None:
        processed_df[target_name] = y.values

    return processed_df