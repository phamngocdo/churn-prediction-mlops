import logging

from churn_mlops.config import RAW_DATA_DIR, TARGET, TEST_SIZE, RANDOM_STATE
from churn_mlops.data.loader import load_raw_data
from churn_mlops.data.splitter import save_splits, split_train_test
from churn_mlops.models.train import evaluate_on_test, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    df = load_raw_data()

    train_df, test_df = split_train_test(
        df, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    save_splits(train_df, test_df, RAW_DATA_DIR)

    pipeline, train_metrics, val_metrics = train_model(train_df)
    test_metrics = evaluate_on_test(pipeline, test_df)

    logger.info("Test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()