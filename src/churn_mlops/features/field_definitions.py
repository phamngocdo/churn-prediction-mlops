"""
Single source of truth for every input field of the churn model.

To add/remove a field from BOTH the FastAPI request schema and the Streamlit
form, edit ONLY this file — `serving/schemas.py` and `serving/streamlit_app.py`
loop over FIELD_SPECS and never need to change.

`value_map` is only needed when the value shown to the user in the UI differs
from the raw value the trained pipeline expects. Example: SeniorCitizen is
stored as int 0/1 in the training data (clean_features casts it to dtype
"object" but keeps the literal 0/1 values), while the form shows "No"/"Yes"
for readability — value_map converts the display value back to what the
model's OneHotEncoder was actually fit on.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    label: str
    options: tuple = ()
    default: object = None
    min_value: float | None = None
    max_value: float | None = None
    value_map: dict | None = None


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec("gender", "categorical", "Gender", options=("Female", "Male"), default="Female"),
    FieldSpec(
        "SeniorCitizen", "binary", "Senior citizen?",
        options=("No", "Yes"), default="No",
        value_map={"No": 0, "Yes": 1},
    ),
    FieldSpec("Partner", "binary", "Has a partner?", options=("No", "Yes"), default="No"),
    FieldSpec("Dependents", "binary", "Has dependents?", options=("No", "Yes"), default="No"),
    FieldSpec("tenure", "numeric", "Tenure (months)", default=12, min_value=0, max_value=100),
    FieldSpec("PhoneService", "binary", "Has phone service?", options=("No", "Yes"), default="Yes"),
    FieldSpec(
        "MultipleLines", "categorical", "Multiple lines",
        options=("No", "Yes", "No phone service"), default="No",
    ),
    FieldSpec(
        "InternetService", "categorical", "Internet service",
        options=("DSL", "Fiber optic", "No"), default="DSL",
    ),
    FieldSpec(
        "OnlineSecurity", "categorical", "Online security",
        options=("No", "Yes", "No internet service"), default="No",
    ),
    FieldSpec(
        "OnlineBackup", "categorical", "Online backup",
        options=("No", "Yes", "No internet service"), default="No",
    ),
    FieldSpec(
        "DeviceProtection", "categorical", "Device protection",
        options=("No", "Yes", "No internet service"), default="No",
    ),
    FieldSpec(
        "TechSupport", "categorical", "Tech support",
        options=("No", "Yes", "No internet service"), default="No",
    ),
    FieldSpec(
        "StreamingTV", "categorical", "Streaming TV",
        options=("No", "Yes", "No internet service"), default="No",
    ),
    FieldSpec(
        "StreamingMovies", "categorical", "Streaming movies",
        options=("No", "Yes", "No internet service"), default="No",
    ),
    FieldSpec(
        "Contract", "categorical", "Contract type",
        options=("Month-to-month", "One year", "Two year"), default="Month-to-month",
    ),
    FieldSpec("PaperlessBilling", "binary", "Paperless billing?", options=("No", "Yes"), default="Yes"),
    FieldSpec(
        "PaymentMethod", "categorical", "Payment method",
        options=(
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ),
        default="Electronic check",
    ),
    FieldSpec("MonthlyCharges", "numeric", "Monthly charges ($)", default=70.0, min_value=0.0, max_value=200.0),
    FieldSpec("TotalCharges", "numeric", "Total charges ($)", default=840.0, min_value=0.0, max_value=10000.0),

    # --- Add new feature ---
    # FieldSpec("NewField", "categorical", "New Field", options=("A", "B"), default="A"),
]
