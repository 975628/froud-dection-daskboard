"""Load the saved model and score new transaction rows for review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_cleaning import PROJECT_ROOT
from src.train_model import MODEL_PATH, prepare_features


def load_model_bundle(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load a locally trained model bundle."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run `python -m src.train_model` first."
        )
    return joblib.load(path)


def score_transactions(
    dataframe: pd.DataFrame,
    bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add experimental probabilities and review risk bands without labels-as-input."""
    model_bundle = bundle if bundle is not None else load_model_bundle()
    features = prepare_features(dataframe)
    required_features = model_bundle["feature_columns"]
    missing = sorted(set(required_features).difference(features.columns))
    if missing:
        raise ValueError(
            "Input transactions are missing model features: " + ", ".join(missing)
        )

    result = dataframe.copy()
    probabilities = model_bundle["pipeline"].predict_proba(
        features[required_features]
    )[:, 1]
    thresholds = model_bundle["thresholds"]
    result["Fraud_Probability"] = probabilities
    result["Model_Flag"] = (
        result["Fraud_Probability"] >= thresholds["decision_threshold"]
    )
    result["Risk_Category"] = pd.cut(
        result["Fraud_Probability"],
        bins=[
            -float("inf"),
            thresholds["medium_risk_probability"],
            thresholds["high_risk_probability"],
            float("inf"),
        ],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype("string")
    return result


if __name__ == "__main__":
    processed = PROJECT_ROOT / "data" / "processed" / "transactions_cleaned.csv"
    if not processed.is_file():
        raise FileNotFoundError("Run `python -m src.data_cleaning` before scoring.")
    rows = pd.read_csv(processed)
    scores = score_transactions(rows)
    print(scores[["Transaction_ID", "Fraud_Probability", "Model_Flag", "Risk_Category"]].head())
