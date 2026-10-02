import io

import pandas as pd
import requests
import streamlit as st

from churn_mlops.config import API_URL
from churn_mlops.features.field_definitions import FIELD_SPECS

st.set_page_config(page_title="Churn Prediction", page_icon="📉")
st.title("📉 Customer Churn Prediction")

tab_form, tab_file = st.tabs(["Single customer (form)", "Batch file upload"])

# ---------------------------------------------------------------------------
# Tab 1: single-record form, auto-generated from FIELD_SPECS
# ---------------------------------------------------------------------------
with tab_form:
    st.caption("Fill in the customer's details and get an instant prediction.")

    with st.form("churn_form"):
        customer_id = st.text_input("Customer ID (optional)", value="")

        values: dict = {}
        for spec in FIELD_SPECS:
            if spec.kind == "numeric":
                values[spec.name] = st.number_input(
                    spec.label,
                    value=float(spec.default),
                    min_value=float(spec.min_value) if spec.min_value is not None else None,
                    max_value=float(spec.max_value) if spec.max_value is not None else None,
                )
            else:  # "binary" or "categorical" -> select box
                options = list(spec.options)
                default_index = options.index(spec.default) if spec.default in options else 0
                values[spec.name] = st.selectbox(spec.label, options, index=default_index)

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {"customerID": customer_id or None, **values}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Request to the API failed: {exc}")
        else:
            is_churn = result["churn_prediction"] == 1
            label = "⚠️ Will churn" if is_churn else "✅ Will stay"
            st.metric(label, f"{result['churn_probability']:.1%} probability of churn")

# ---------------------------------------------------------------------------
# Tab 2: batch scoring via file upload
# ---------------------------------------------------------------------------
with tab_file:
    st.caption("Upload a CSV, JSON, or Parquet file of customers to score them all at once.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "json", "parquet"])

    if uploaded_file is not None:
        suffix = uploaded_file.name.split(".")[-1].lower()
        input_df = None
        try:
            if suffix == "csv":
                input_df = pd.read_csv(uploaded_file)
            elif suffix == "json":
                input_df = pd.read_json(uploaded_file)
            else:  # parquet
                input_df = pd.read_parquet(uploaded_file)
        except Exception as exc:  # noqa: BLE001 - surfaced directly to the user
            st.error(f"Could not read the uploaded file: {exc}")

        if input_df is not None:
            st.write(f"Loaded {len(input_df)} rows. Preview:")
            st.dataframe(input_df.head())

            if st.button("Score file"):
                # NaN is not valid JSON -> convert to None before sending
                records = input_df.where(pd.notnull(input_df), None).to_dict(orient="records")
                try:
                    response = requests.post(
                        f"{API_URL}/predict_batch",
                        json={"records": records},
                        timeout=60,
                    )
                    response.raise_for_status()
                    results = pd.DataFrame(response.json())
                except requests.RequestException as exc:
                    st.error(f"Request to the API failed: {exc}")
                else:
                    output_df = pd.concat(
                        [
                            input_df.reset_index(drop=True),
                            results[["churn_prediction", "churn_probability"]],
                        ],
                        axis=1,
                    )
                    st.success("Done! Preview of results:")
                    st.dataframe(output_df.head())

                    buffer = io.BytesIO()
                    output_df.to_csv(buffer, index=False)
                    st.download_button(
                        "Download results as CSV",
                        data=buffer.getvalue(),
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                    )
