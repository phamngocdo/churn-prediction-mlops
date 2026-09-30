import pandas as pd

from churn_mlops.data.splitter import save_splits, split_train_test


def test_split_train_test_keeps_target_distribution():
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(1, 21)],
            "TotalCharges": [100 + i for i in range(20)],
            "Churn": ["Yes", "No"] * 10,
        }
    )

    train_df, test_df = split_train_test(
        df=df,
        target="Churn",
        test_size=0.2,
        random_state=42,
    )

    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df.columns) == set(df.columns)
    assert set(test_df.columns) == set(df.columns)
    assert abs(train_df["Churn"].eq("Yes").mean() - 0.5) < 0.25
    assert abs(test_df["Churn"].eq("Yes").mean() - 0.5) < 0.25


def test_save_splits_writes_csv_files(tmp_path):
    train_df = pd.DataFrame({"customerID": ["C1", "C2"], "Churn": ["No", "Yes"]})
    test_df = pd.DataFrame({"customerID": ["C3"], "Churn": ["Yes"]})

    save_splits(train_df, test_df, str(tmp_path / "splits"))

    assert (tmp_path / "splits" / "train.csv").exists()
    assert (tmp_path / "splits" / "test.csv").exists()

    saved_train = pd.read_csv(tmp_path / "splits" / "train.csv")
    saved_test = pd.read_csv(tmp_path / "splits" / "test.csv")

    assert list(saved_train.columns) == ["customerID", "Churn"]
    assert list(saved_test.columns) == ["customerID", "Churn"]
