import logging
import os
from pathlib import Path

import kagglehub
import pandas as pd

from churn_mlops.config import KAGGLE_DATASET_SLUG

logger = logging.getLogger(__name__)


def download_dataset() -> str:
    """Download the dataset from Kaggle Hub, return the local (cached) directory path."""
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
    logger.info("Dataset downloaded to %s", dataset_path)
    return dataset_path


def load_raw_data(dataset_dir: str | None = None) -> pd.DataFrame:
    """Read the first CSV file found in the dataset directory into a DataFrame.

    Args:
        dataset_dir: directory containing the CSV. If None, auto-download from Kaggle.
    """
    if dataset_dir is None:
        dataset_dir = download_dataset()

    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {dataset_dir}")

    file_path = Path(dataset_dir) / csv_files[0]
    df = pd.read_csv(file_path)
    logger.info("Loaded dataset '%s' with shape %s", file_path.name, df.shape)
    return df