from churn_mlops.config import RAW_DATA_DIR, TARGET, TEST_SIZE, RANDOM_STATE
from churn_mlops.data.loader import load_raw_data
from churn_mlops.data.splitter import save_splits, split_train_test
from churn_mlops.models.train import train_model

def main() -> None:
    df = load_raw_data()

    train_df, test_df = split_train_test(
        df, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    save_splits(train_df, test_df, RAW_DATA_DIR)

    train_model(train_df, test_df=test_df)


if __name__ == "__main__":
    main()