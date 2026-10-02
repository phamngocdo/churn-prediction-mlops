from mlflow.tracking import MlflowClient

def get_latest_model_uri(model_name: str) -> str:
    """Return the models:/ URI of the highest version number registered for
    `model_name` — i.e. whatever was registered most recently, regardless of
    stage/alias. This keeps serving code decoupled from how staging/promotion
    is managed (MLflow has deprecated stages in favor of aliases, so we avoid
    depending on either).

    Raises:
        ValueError: if no version of `model_name` has been registered yet.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(
            f"No versions found for registered model '{model_name}'. "
            "Has a training run registered it yet?"
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{model_name}/{latest.version}"
