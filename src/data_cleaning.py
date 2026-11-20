"""Validate and clean the supplied transaction dataset without overwriting raw data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_fraud_dataset1.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_cleaned.csv"
PROFILE_PATH = PROJECT_ROOT / "reports" / "data_profile.json"
TARGET_COLUMN = "Fraud_Label"


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a CSV and give a clear error when its path is wrong."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Put the CSV in data/raw/ and try again."
        )
    return pd.read_csv(path)


def clean_transactions(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize labels, dates, and text while recording every row-level change.

    Missing feature values are intentionally retained here. The model pipeline
    imputes them after the chronological split, using training data only.
    """
    if dataframe.empty:
        raise ValueError("The input dataset has no rows.")

    cleaned = dataframe.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    if cleaned.columns.duplicated().any():
        duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicates}")

    required = {TARGET_COLUMN, "Transaction_Amount", "Date"}
    missing_columns = sorted(required.difference(cleaned.columns))
    if missing_columns:
        raise ValueError(
            "The supplied dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        values = cleaned[column].astype("string").str.strip()
        cleaned[column] = values.mask(values.eq(""), pd.NA)

    numeric_fields = [
        "Transaction_Amount",
        "Account_Balance",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Age",
    ]
    numeric_conversion_failures: dict[str, int] = {}
    for column in numeric_fields:
        if column in cleaned.columns:
            original = cleaned[column]
            converted = pd.to_numeric(original, errors="coerce")
            numeric_conversion_failures[column] = int(
                (original.notna() & converted.isna()).sum()
            )
            cleaned[column] = converted

    original_labels = cleaned[TARGET_COLUMN]
    numeric_labels = pd.to_numeric(original_labels, errors="coerce")
    missing_label_rows = int(numeric_labels.isna().sum())
    cleaned = cleaned.loc[numeric_labels.notna()].copy()
    numeric_labels = numeric_labels.loc[numeric_labels.notna()]

    invalid_labels = sorted(set(numeric_labels.unique()).difference({0, 1}))
    if invalid_labels:
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0 and 1; found {invalid_labels}."
        )
    cleaned[TARGET_COLUMN] = numeric_labels.astype("int8")

    parsed_dates = pd.to_datetime(cleaned["Date"], errors="coerce", dayfirst=True)
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        examples = cleaned.loc[parsed_dates.isna(), "Date"].head(5).tolist()
        raise ValueError(
            f"Could not parse {invalid_dates} transaction dates. Examples: {examples}"
        )
    cleaned["Date"] = parsed_dates

    negative_amounts = int(
        (pd.to_numeric(cleaned["Transaction_Amount"], errors="coerce") < 0).sum()
    )
    negative_card_ages = 0
    if "Card_Age" in cleaned.columns:
        negative_card_ages = int(
            (pd.to_numeric(cleaned["Card_Age"], errors="coerce") < 0).sum()
        )

    audit = {
        "rows_before_cleaning": int(len(dataframe)),
        "rows_after_cleaning": int(len(cleaned)),
        "duplicate_rows_removed": duplicate_rows,
        "rows_without_valid_target_removed": missing_label_rows,
        "missing_values_by_column": {
            column: int(count)
            for column, count in cleaned.isna().sum().items()
            if count
        },
        "unparseable_numeric_values_by_column": {
            column: count
            for column, count in numeric_conversion_failures.items()
            if count
        },
        "negative_transaction_amounts": negative_amounts,
        "negative_card_ages": negative_card_ages,
        "date_min": cleaned["Date"].min().date().isoformat(),
        "date_max": cleaned["Date"].max().date().isoformat(),
        "target_distribution": {
            str(label): int(count)
            for label, count in cleaned[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }
    return cleaned.reset_index(drop=True), audit


def profile_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-safe summary for student-friendly documentation."""
    profile: dict[str, Any] = {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "column_details": [
            {
                "name": column,
                "dtype": str(dataframe[column].dtype),
                "missing": int(dataframe[column].isna().sum()),
                "unique": int(dataframe[column].nunique(dropna=False)),
            }
            for column in dataframe.columns
        ],
    }
    if TARGET_COLUMN in dataframe.columns:
        counts = dataframe[TARGET_COLUMN].value_counts(dropna=False).sort_index()
        profile["target_column"] = TARGET_COLUMN
        profile["target_counts"] = {
            str(label): int(count) for label, count in counts.items()
        }
        profile["fraud_rate"] = float(dataframe[TARGET_COLUMN].mean())
    if "Date" in dataframe.columns and not dataframe["Date"].isna().all():
        profile["date_min"] = dataframe["Date"].min().date().isoformat()
        profile["date_max"] = dataframe["Date"].max().date().isoformat()
    return profile


def run_cleaning() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the processed CSV and audit report while preserving the original."""
    cleaned, audit = clean_transactions(load_dataset())
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = cleaned.copy()
    output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")
    output.to_csv(PROCESSED_DATA_PATH, index=False)
    PROFILE_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return cleaned, audit


if __name__ == "__main__":
    _, cleaning_audit = run_cleaning()
    print(json.dumps(cleaning_audit, indent=2))
