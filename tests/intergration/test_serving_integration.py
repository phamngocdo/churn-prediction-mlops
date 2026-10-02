from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from churn_mlops.config import ID_COLUMN, MLFLOW_TRACKING_URI
from churn_mlops.features.field_definitions import FIELD_SPECS
from churn_mlops.serving import api


class RecordingModel:
    def __init__(self):
        self.feature_batches = []

    def predict(self, features):
        self.feature_batches.append(features.copy())
        return (features["tenure"].to_numpy() >= 24).astype(int)

    def predict_proba(self, features):
        churn = (features["tenure"].to_numpy() >= 24).astype(float)
        return np.column_stack((1 - np.where(churn, 0.8, 0.2), np.where(churn, 0.8, 0.2)))


@pytest.fixture
def serving_client(monkeypatch) -> Iterator[tuple[TestClient, RecordingModel, dict]]:
    model = RecordingModel()
    captured = {}

    def fake_get_latest_model_uri(model_name):
        captured["model_name"] = model_name
        return f"models:/{model_name}/5"

    def fake_load_model(model_uri):
        captured["model_uri"] = model_uri
        return model

    monkeypatch.setattr(api, "get_latest_model_uri", fake_get_latest_model_uri)
    monkeypatch.setattr(
        api.mlflow,
        "set_tracking_uri",
        lambda uri: captured.setdefault("set_tracking_uri", uri),
    )
    monkeypatch.setattr(api.mlflow.sklearn, "load_model", fake_load_model)

    previous_model, previous_uri = api._model, api._model_uri
    try:
        with TestClient(api.app) as client:
            yield client, model, captured
    finally:
        api._model, api._model_uri = previous_model, previous_uri


def make_record(customer_id: str, tenure: float) -> dict:
    record = {spec.name: spec.default for spec in FIELD_SPECS}
    record.update({"customerID": customer_id, "SeniorCitizen": "Yes", "tenure": tenure})
    return record


def test_api_startup_single_and_batch_prediction(serving_client):
    client, model, captured = serving_client

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_uri": "models:/churn_model/5"}
    assert captured["model_name"] == "churn_model"
    assert captured["model_uri"] == "models:/churn_model/5"
    assert captured["set_tracking_uri"] == MLFLOW_TRACKING_URI

    single = client.post("/predict", json=make_record("C-1", tenure=30))
    assert single.status_code == 200
    assert single.json() == {
        "customerID": "C-1",
        "churn_prediction": 1,
        "churn_probability": 0.8,
    }

    batch = client.post(
        "/predict_batch",
        json={
            "records": [
                make_record("C-2", tenure=10),
                make_record("C-3", tenure=40),
            ]
        },
    )
    assert batch.status_code == 200
    assert batch.json() == [
        {"customerID": "C-2", "churn_prediction": 0, "churn_probability": 0.2},
        {"customerID": "C-3", "churn_prediction": 1, "churn_probability": 0.8},
    ]

    single_features, batch_features = model.feature_batches
    assert ID_COLUMN not in single_features.columns
    assert ID_COLUMN not in batch_features.columns
    assert single_features["SeniorCitizen"].tolist() == [1]
    assert single_features["SeniorCitizen"].dtype == object
    assert single_features["TotalCharges"].dtype.kind == "f"


def test_api_reports_not_ready_and_rejects_invalid_payload(serving_client):
    client, _, _ = serving_client

    api._model = None
    response = client.post("/predict", json=make_record("C-4", tenure=10))
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded yet"

    api._model = RecordingModel()
    invalid = make_record("C-5", tenure=10)
    invalid["tenure"] = "not-a-number"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def install_frontend_api_bridge(monkeypatch, client):
    calls = []

    def post(url, json, timeout):
        path = url.removeprefix("http://localhost:8000")
        calls.append((path, json, timeout))
        api_response = client.post(path, json=json)
        response = requests.Response()
        response.status_code = api_response.status_code
        response._content = api_response.content
        response.url = url
        return response

    monkeypatch.setattr(requests, "post", post)
    return calls


def test_streamlit_single_customer_form_calls_backend(serving_client, monkeypatch):
    client, _, _ = serving_client
    calls = install_frontend_api_bridge(monkeypatch, client)
    app_path = Path(__file__).resolve().parents[2] / "src/churn_mlops/serving/streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run()
    app.text_input[0].set_value("UI-100")
    app.number_input[0].set_value(30.0)
    app.button[0].click().run()

    assert not app.exception
    assert calls[0][0] == "/predict"
    assert calls[0][1]["customerID"] == "UI-100"
    assert calls[0][1]["SeniorCitizen"] == "No"
    assert app.metric[0].label == "⚠️ Will churn"
    assert app.metric[0].value == "80.0% probability of churn"


def test_streamlit_csv_upload_scores_and_exposes_download(serving_client, monkeypatch):
    client, _, _ = serving_client
    calls = install_frontend_api_bridge(monkeypatch, client)
    app_path = Path(__file__).resolve().parents[2] / "src/churn_mlops/serving/streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run()

    records = pd.DataFrame(
        [make_record("UI-1", 10), make_record("UI-2", 40)]
    ).drop(columns=["customerID"])
    csv_bytes = records.to_csv(index=False).encode("utf-8")
    app.file_uploader[0].set_value(("customers.csv", csv_bytes, "text/csv")).run()
    app.button[1].click().run()

    assert not app.exception
    assert calls[0][0] == "/predict_batch"
    assert len(calls[0][1]["records"]) == 2
    assert app.success[0].value == "Done! Preview of results:"
    scored = app.dataframe[1].value
    assert {"churn_prediction", "churn_probability"}.issubset(scored.columns)
    assert scored["churn_prediction"].tolist() == [0, 1]
