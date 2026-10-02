import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from churn_mlops.config import ID_COLUMN, MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME
from churn_mlops.features.preprocessing import clean_features
from churn_mlops.models.registry import get_latest_model_uri

logger = logging.getLogger("churn_mlops.batch_scoring")

OUTPUT_SCHEMA = StructType(
    [
        StructField(ID_COLUMN, StringType(), True),
        StructField("churn_prediction", IntegerType(), True),
        StructField("churn_probability", DoubleType(), True),
    ]
)


def predict_batch_spark(
    spark: SparkSession,
    df: DataFrame,
    model_name: str = MODEL_REGISTRY_NAME,
) -> DataFrame:
    """Score a Spark DataFrame of raw customer records with the latest MLflow model.

    Args:
        spark: active SparkSession.
        df: Spark DataFrame with the same raw columns as the training data
            (the target column is not required and is ignored if present).
        model_name: registered MLflow model name. Defaults to the latest
            registered version (see `models/registry.py`) — no stage/alias
            needs to be set manually.

    Returns:
        Spark DataFrame with columns:
            [ID_COLUMN, "churn_prediction" (0/1), "churn_probability" (0-1)]
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = get_latest_model_uri(model_name)
    logger.info("Batch scoring with model: %s", model_uri)

    # Broadcast only the (tiny) URI string, not the model object itself --
    # each executor resolves and loads its own local copy on first use.
    model_uri_bc = spark.sparkContext.broadcast(model_uri)

    def score_partitions(iterator):
        # Imports happen inside the worker process (executors are separate
        # Python processes and don't inherit the driver's imports).
        import mlflow.sklearn

        model = None  # lazy-loaded once per partition/executor task
        for pdf in iterator:
            if model is None:
                model = mlflow.sklearn.load_model(model_uri_bc.value)

            if ID_COLUMN in pdf.columns:
                ids = pdf[ID_COLUMN].astype(str)
            else:
                ids = pd.Series([str(i) for i in range(len(pdf))])

            features = clean_features(pdf)
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]

            yield pd.DataFrame(
                {
                    ID_COLUMN: ids.values,
                    "churn_prediction": predictions.astype(int),
                    "churn_probability": probabilities.astype(float),
                }
            )

    return df.mapInPandas(score_partitions, schema=OUTPUT_SCHEMA)
