import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import clean_transactions
from src.predict import score_transactions
from src.train_model import chronological_split, prepare_features


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Transaction_ID": ["TXN-1", "TXN-2", "TXN-3", "TXN-4"],
            "User_ID": ["U1", "U2", "U3", "U4"],
            "Transaction_Amount": [10.0, 20.0, 30.0, 40.0],
            "Transaction_Type": ["POS", "Online", "POS", "Online"],
            "Date": ["01 January 2023", "02 January 2023", "03 January 2023", "04 January 2023"],
            "Previous_Fraudulent_Activity": [0, 0, 1, 0],
            "Fraud_Label": [0, 1, 0, 1],
        }
    )


def test_cleaning_parses_dates_and_records_duplicate_removal() -> None:
    data = pd.concat([sample_transactions(), sample_transactions().iloc[[0]]], ignore_index=True)

    cleaned, audit = clean_transactions(data)

    assert len(cleaned) == 4
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
    assert audit["duplicate_rows_removed"] == 1
    assert audit["target_distribution"] == {"0": 2, "1": 2}


def test_cleaning_rejects_non_binary_target() -> None:
    data = sample_transactions()
    data.loc[0, "Fraud_Label"] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        clean_transactions(data)


def test_feature_builder_drops_identifiers_and_adds_calendar_features() -> None:
    features = prepare_features(sample_transactions())

    assert "Transaction_ID" not in features
    assert "User_ID" not in features
    assert "Fraud_Label" not in features
    assert "Transaction_Month" in features
    assert "Transaction_Weekday" in features


def test_chronological_split_keeps_later_dates_in_test() -> None:
    data = sample_transactions()
    train, test, cutoff = chronological_split(data, test_fraction=0.25)

    assert cutoff == "2023-01-04"
    assert pd.to_datetime(data.loc[train, "Date"]).max() < pd.to_datetime(
        data.loc[test, "Date"]
    ).min()


def test_risk_bands_are_explicit_and_ordered() -> None:
    class FixedProbabilityModel:
        def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
            return np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9], [0.7, 0.3]])

    bundle = {
        "pipeline": FixedProbabilityModel(),
        "feature_columns": [
            "Transaction_Amount",
            "Transaction_Type",
            "Previous_Fraudulent_Activity",
            "Transaction_Month",
            "Transaction_Weekday",
        ],
        "thresholds": {
            "decision_threshold": 0.5,
            "medium_risk_probability": 0.3,
            "high_risk_probability": 0.65,
        },
    }
    scores = score_transactions(sample_transactions(), bundle)

    assert scores["Risk_Category"].tolist() == ["Low", "Medium", "High", "Medium"]
    assert scores["Model_Flag"].tolist() == [False, True, True, False]
