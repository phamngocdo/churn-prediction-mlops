import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn_mlops.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_NAME,
    RANDOM_STATE,
    TARGET,
    VAL_SIZE,
)
from churn_mlops.features.preprocessing import (
    build_preprocessor,
    clean_features,
    encode_target,
    get_feature_types,
)
from churn_mlops.models.eval import compute_metrics

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
}


def build_pipeline(
    categorical_features: list[str],
    numerical_features: list[str],
    params: dict | None = None,
) -> Pipeline:
    """Combine preprocessing + model into a single Pipeline object.

    This is the key point for production: pipeline.predict(raw_df) runs
    directly on raw data (after clean_features), with no separate
    transform step maintained elsewhere — reducing the risk of drift
    between training and serving.
    """
    params = params or DEFAULT_PARAMS
    preprocessor = build_preprocessor(categorical_features, numerical_features)
    model = XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric="logloss")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(
    train_df: pd.DataFrame,
    params: dict | None = None,
    register_model: bool = True,
    test_df: pd.DataFrame | None = None,
) -> tuple[Pipeline, dict, dict]:
    """Train the XGBoost pipeline, log everything to MLflow, return the fitted pipeline.

    Args:
        train_df: dataframe containing both features and the target column
            (not yet split into a validation set).
        params: hyperparameters for XGBClassifier, defaults to DEFAULT_PARAMS.
        register_model: if True, register the model in the MLflow Model Registry.
        test_df: optional held-out test dataframe whose metrics are logged to the
            same MLflow run as the train and validation metrics.

    Returns:
        (fitted_pipeline, train_metrics, val_metrics)
    """
    params = params or DEFAULT_PARAMS

    X = clean_features(train_df.drop(columns=[TARGET]))
    y = encode_target(train_df[TARGET])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    categorical_features, numerical_features = get_feature_types(X_train)
    pipeline = build_pipeline(categorical_features, numerical_features, params)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_val_samples", len(X_val))

        pipeline.fit(X_train, y_train)

        train_metrics = compute_metrics(y_train, pipeline.predict(X_train))
        val_metrics = compute_metrics(y_val, pipeline.predict(X_val))

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        if test_df is not None:
            test_metrics = evaluate_on_test(pipeline, test_df)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            logger.info("Test metrics: %s", test_metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME if register_model else None,
            serialization_format="skops",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.compose._column_transformer._RemainderColsList",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        logger.info("Train metrics: %s", train_metrics)
        logger.info("Validation metrics: %s", val_metrics)

    return pipeline, train_metrics, val_metrics


def evaluate_on_test(pipeline: Pipeline, test_df: pd.DataFrame) -> dict:
    """Evaluate the fitted pipeline on the held-out test set (never used during training)."""
    X_test = clean_features(test_df.drop(columns=[TARGET]))
    y_test = encode_target(test_df[TARGET])
    y_pred = pipeline.predict(X_test)
    return compute_metrics(y_test, y_pred)