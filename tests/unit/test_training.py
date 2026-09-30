import pandas as pd

from churn_mlops.data.loader import download_dataset, load_raw_data
from churn_mlops.features.preprocessing import (
    build_preprocessor,
    clean_features,
    encode_target,
    get_feature_types,
    transform_to_dataframe,
)
from churn_mlops.models.train import build_pipeline, evaluate_on_test, train_model


def build_churn_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(1, 25)],
            "gender": ["Female", "Male"] * 12,
            "SeniorCitizen": [0, 1] * 12,
            "TotalCharges": [100.0 + i for i in range(24)],
            "InternetService": ["DSL", "Fiber", "DSL", "Fiber"] * 6,
            "Churn": ["No", "Yes"] * 12,
        }
    )


def test_clean_features_and_feature_types():
    raw_df = build_churn_dataset()

    cleaned = clean_features(raw_df.drop(columns=["Churn"]))

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].notna().all()
    assert cleaned["SeniorCitizen"].dtype == object

    categorical_features, numerical_features = get_feature_types(cleaned)

    assert "gender" in categorical_features
    assert "InternetService" in categorical_features
    assert "SeniorCitizen" in categorical_features
    assert "TotalCharges" in numerical_features


def test_build_preprocessor_and_pipeline_fit():
    raw_df = build_churn_dataset()
    X = clean_features(raw_df.drop(columns=["Churn"]))
    y = encode_target(raw_df["Churn"])

    categorical_features, numerical_features = get_feature_types(X)
    preprocessor = build_preprocessor(categorical_features, numerical_features)
    pipeline = build_pipeline(
        categorical_features,
        numerical_features,
        params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
    )

    assert preprocessor is not None
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert set(preds.tolist()) <= {0, 1}


def test_train_model_runs_and_logs_metrics(monkeypatch):
    df = build_churn_dataset()
    captured = {}

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_log_param(name, value):
        captured.setdefault("params", {})[name] = value

    def fake_log_metrics(metrics):
        captured.setdefault("metrics", {}).update(metrics)

    def fake_log_model(sk_model, artifact_path, registered_model_name=None):
        captured["artifact_path"] = artifact_path
        captured["registered_model_name"] = registered_model_name

    monkeypatch.setattr("churn_mlops.models.train.mlflow.set_tracking_uri", lambda uri: captured.setdefault("uri", uri))
    monkeypatch.setattr("churn_mlops.models.train.mlflow.set_experiment", lambda name: captured.setdefault("experiment", name))
    monkeypatch.setattr("churn_mlops.models.train.mlflow.start_run", lambda: DummyRun())
    monkeypatch.setattr("churn_mlops.models.train.mlflow.log_params", lambda params: captured.setdefault("logged_params", {}).update(params))
    monkeypatch.setattr("churn_mlops.models.train.mlflow.log_param", fake_log_param)
    monkeypatch.setattr("churn_mlops.models.train.mlflow.log_metrics", fake_log_metrics)
    monkeypatch.setattr("churn_mlops.models.train.mlflow.sklearn.log_model", fake_log_model)

    pipeline, train_metrics, val_metrics = train_model(
        df,
        params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
        register_model=False,
    )

    assert pipeline is not None
    assert set(train_metrics).issubset({"accuracy", "precision", "recall", "f1"})
    assert set(val_metrics).issubset({"accuracy", "precision", "recall", "f1"})
    assert captured["artifact_path"] == "model"
    assert captured["registered_model_name"] is None
    assert "train_accuracy" in captured["metrics"]
    assert "val_accuracy" in captured["metrics"]


def test_transform_to_dataframe_includes_feature_names_and_target():
    df = build_churn_dataset()
    X = clean_features(df.drop(columns=["Churn"]))
    y = encode_target(df["Churn"])

    categorical_features, numerical_features = get_feature_types(X)
    preprocessor = build_preprocessor(categorical_features, numerical_features)
    preprocessor.fit(X)

    transformed = transform_to_dataframe(preprocessor, X, y)

    assert transformed.shape[0] == len(X)
    assert "Churn" in transformed.columns
    assert transformed["Churn"].nunique() == 2


def test_transform_to_dataframe_without_target_keeps_feature_columns_only():
    df = build_churn_dataset()
    X = clean_features(df.drop(columns=["Churn"]))

    categorical_features, numerical_features = get_feature_types(X)
    preprocessor = build_preprocessor(categorical_features, numerical_features)
    preprocessor.fit(X)

    transformed = transform_to_dataframe(preprocessor, X)

    assert "Churn" not in transformed.columns
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0


def test_evaluate_on_test_returns_metric_dict():
    df = build_churn_dataset()
    pipeline = build_pipeline(
        ["gender", "InternetService", "SeniorCitizen"],
        ["TotalCharges"],
        params={"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
    )

    X = clean_features(df.drop(columns=["Churn"]))
    y = encode_target(df["Churn"])
    pipeline.fit(X, y)

    metrics = evaluate_on_test(pipeline, df)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert 0 <= metrics["accuracy"] <= 1


def test_download_dataset_and_load_raw_data(monkeypatch, tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"customerID": ["C1"], "Churn": ["Yes"]}).to_csv(csv_path, index=False)

    monkeypatch.setattr("churn_mlops.data.loader.kagglehub.dataset_download", lambda slug: str(tmp_path))
    assert download_dataset() == str(tmp_path)

    loaded = load_raw_data()
    assert list(loaded.columns) == ["customerID", "Churn"]
    assert loaded.iloc[0]["Churn"] == "Yes"


def test_load_raw_data_raises_when_no_csv_exists(tmp_path):
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()

    try:
        load_raw_data(str(empty_dir))
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
