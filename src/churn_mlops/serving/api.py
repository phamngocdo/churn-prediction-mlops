import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_mlops.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME
from churn_mlops.features.field_definitions import FIELD_SPECS
from churn_mlops.features.preprocessing import clean_features
from churn_mlops.models.registry import get_latest_model_uri
from churn_mlops.serving.schemas import BatchPredictionRequest, CustomerRecord, PredictionResponse

logger = logging.getLogger("churn_mlops.api")

_model = None
_model_uri: str | None = None


def load_model() -> None:
    """Load the latest registered model version once, at process startup."""
    global _model, _model_uri
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model_uri = get_latest_model_uri(MODEL_REGISTRY_NAME)
    _model = mlflow.sklearn.load_model(_model_uri)
    logger.info("Loaded model from %s", _model_uri)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _model is not None else "loading", "model_uri": _model_uri}


def _record_to_row(record: "CustomerRecord") -> dict: # type: ignore
    """Apply each field's value_map (e.g. SeniorCitizen 'Yes'/'No' -> 1/0) so
    the row matches the exact representation the model was trained on."""
    data = record.model_dump()
    row = {}
    for spec in FIELD_SPECS:
        value = data.get(spec.name)
        if spec.value_map is not None and value in spec.value_map:
            value = spec.value_map[value]
        row[spec.name] = value
    return row


def _predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    features = clean_features(df)
    predictions = _model.predict(features)
    probabilities = _model.predict_proba(features)[:, 1]
    return pd.DataFrame({"churn_prediction": predictions, "churn_probability": probabilities})


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord) -> PredictionResponse:  # type: ignore[valid-type]
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    row = _record_to_row(record)
    df = pd.DataFrame([row])
    result = _predict_dataframe(df)

    return PredictionResponse(
        customerID=record.customerID,
        churn_prediction=int(result.loc[0, "churn_prediction"]),
        churn_probability=float(result.loc[0, "churn_probability"]),
    )


@app.post("/predict_batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchPredictionRequest) -> list[PredictionResponse]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    rows = [_record_to_row(r) for r in request.records]
    ids = [r.customerID for r in request.records]
    df = pd.DataFrame(rows)
    result = _predict_dataframe(df)

    return [
        PredictionResponse(
            customerID=ids[i],
            churn_prediction=int(result.loc[i, "churn_prediction"]),
            churn_probability=float(result.loc[i, "churn_probability"]),
        )
        for i in range(len(result))
    ]
