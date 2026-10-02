
import os

# --- Data ---
TARGET: str = "Churn"
ID_COLUMN: str = "customerID"
RANDOM_STATE: int = 42

KAGGLE_DATASET_SLUG: str = "blastchar/telco-customer-churn"
RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")

# --- Train/val/test split ---
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.2

# --- MLflow ---
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
MODEL_REGISTRY_NAME: str = os.getenv("MODEL_REGISTRY_NAME", "churn_model")

# --- Serving ---------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", "http://localhost:8000")