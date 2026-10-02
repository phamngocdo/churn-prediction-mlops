from typing import Optional

from pydantic import BaseModel, Field, create_model

from churn_mlops.features.field_definitions import FIELD_SPECS


def _python_type(kind: str):
    return float if kind == "numeric" else str


_dynamic_fields = {
    spec.name: (_python_type(spec.kind), Field(default=spec.default, description=spec.label))
    for spec in FIELD_SPECS
}
# customerID is not a model feature (it's dropped by clean_features) but is
# accepted so it can be echoed back in the response for correlation.
_dynamic_fields["customerID"] = (
    Optional[str],
    Field(default=None, description="Customer ID, passed through to the response"),
)

CustomerRecord = create_model("CustomerRecord", **_dynamic_fields)  # type: ignore[call-overload]


class PredictionResponse(BaseModel):
    customerID: Optional[str] = None
    churn_prediction: int = Field(description="1 = will churn, 0 = will not churn")
    churn_probability: float = Field(description="Predicted probability of churn (0-1)")


class BatchPredictionRequest(BaseModel):
    records: list[CustomerRecord]  # type: ignore[valid-type]
