from types import SimpleNamespace

import numpy as np
import pandas as pd

from churn_mlops.config import ID_COLUMN, MLFLOW_TRACKING_URI
from churn_mlops.pipelines import batch_scoring_pipeline as pipeline


class FakeSparkDataFrame:
    def __init__(self, batches):
        self.batches = batches
        self.output_schema = None
        self.output_batches = None

    def mapInPandas(self, function, schema):
        self.output_schema = schema
        self.output_batches = list(function(iter(self.batches)))
        return self.output_batches


def test_predict_batch_spark_scores_batches_and_broadcasts_model_uri(monkeypatch):
    loaded_uris = []
    tracking_uris = []
    cleaned_batches = []

    class FakeModel:
        def predict(self, features):
            return np.arange(len(features)) % 2

        def predict_proba(self, features):
            positive = np.linspace(0.25, 0.75, len(features))
            return np.column_stack((1 - positive, positive))

    def fake_load_model(model_uri):
        loaded_uris.append(model_uri)
        return FakeModel()

    def fake_clean_features(frame):
        cleaned_batches.append(frame.copy())
        return frame.drop(columns=[ID_COLUMN], errors="ignore")

    monkeypatch.setattr(pipeline.mlflow, "set_tracking_uri", tracking_uris.append)
    monkeypatch.setattr(pipeline, "get_latest_model_uri", lambda name: f"models:/{name}/7")
    monkeypatch.setattr(pipeline.mlflow.sklearn, "load_model", fake_load_model)
    monkeypatch.setattr(pipeline, "clean_features", fake_clean_features)

    class FakeSparkContext:
        def broadcast(self, value):
            return SimpleNamespace(value=value)

    spark = SimpleNamespace(sparkContext=FakeSparkContext())
    frame = FakeSparkDataFrame(
        [
            pd.DataFrame({ID_COLUMN: [101, "C2"], "tenure": [3, 8]}),
            pd.DataFrame({"tenure": [12, 20]}),
        ]
    )

    result = pipeline.predict_batch_spark(spark, frame, model_name="test_model")

    assert tracking_uris == [MLFLOW_TRACKING_URI]
    assert loaded_uris == ["models:/test_model/7"]
    assert len(cleaned_batches) == 2
    assert frame.output_schema == pipeline.OUTPUT_SCHEMA
    assert result[0][ID_COLUMN].tolist() == ["101", "C2"]
    assert result[1][ID_COLUMN].tolist() == ["0", "1"]
    assert result[0]["churn_prediction"].tolist() == [0, 1]
    assert result[1]["churn_prediction"].tolist() == [0, 1]
    assert result[0]["churn_probability"].tolist() == [0.25, 0.75]
    assert result[1]["churn_probability"].tolist() == [0.25, 0.75]
    assert result[0]["churn_prediction"].dtype.kind in "iu"
    assert result[0]["churn_probability"].dtype.kind == "f"
