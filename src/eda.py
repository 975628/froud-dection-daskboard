"""Generate reproducible EDA charts and data-supported business summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.data_cleaning import (
    PROJECT_ROOT,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    clean_transactions,
    load_dataset,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
INSIGHTS_PATH = PROJECT_ROOT / "reports" / "eda_insights.json"


def load_cleaned_data() -> pd.DataFrame:
    """Load the processed table, creating it from raw data if needed."""
    if PROCESSED_DATA_PATH.exists():
        dataframe = pd.read_csv(PROCESSED_DATA_PATH)
        dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
        return dataframe
    dataframe, _ = clean_transactions(load_dataset())
    return dataframe


def _save_figure(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_eda(dataframe: pd.DataFrame | None = None) -> dict[str, Any]:
    """Save class, amount, category, time, missingness, and correlation charts."""
    frame = (dataframe if dataframe is not None else load_cleaned_data()).copy()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Fraud_Status"] = frame[TARGET_COLUMN].map(
        {0: "Legitimate (0)", 1: "Fraud label (1)"}
    )

    colors = {"Legitimate (0)": "#38bdf8", "Fraud label (1)": "#f43f5e"}

    figure, axis = plt.subplots(figsize=(7, 4))
    counts = frame["Fraud_Status"].value_counts().reindex(colors.keys(), fill_value=0)
    axis.bar(counts.index, counts.values, color=[colors[label] for label in counts.index])
    axis.set(title="Observed target labels", ylabel="Transactions")
    _save_figure(figure, "class_distribution.png")

    figure, axis = plt.subplots(figsize=(8, 4))
    for label, group in frame.groupby("Fraud_Status", observed=True):
        axis.hist(
            group["Transaction_Amount"],
            bins=50,
            alpha=0.55,
            color=colors.get(label, "#a78bfa"),
            label=label,
        )
    axis.set(title="Transaction amount by recorded label", xlabel="Amount (dataset units)")
    axis.legend()
    _save_figure(figure, "transaction_amounts.png")

    category_column = "Transaction_Type"
    if category_column in frame.columns:
        rates = (
            frame.groupby(category_column, observed=True)[TARGET_COLUMN]
            .mean()
            .sort_values(ascending=False)
        )
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(rates.index.astype(str), rates.values, color="#8b5cf6")
        axis.set(
            title="Observed fraud-label rate by transaction type",
            ylabel="Fraud-label rate",
        )
        axis.tick_params(axis="x", rotation=20)
        _save_figure(figure, "fraud_rate_by_type.png")

    daily = (
        frame.dropna(subset=["Date"])
        .groupby("Date", observed=True)[TARGET_COLUMN]
        .agg(["size", "sum"])
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(daily.index, daily["size"], color="#38bdf8", linewidth=1.2)
    axis.set(title="Daily transaction volume", ylabel="Transactions")
    _save_figure(figure, "daily_transaction_volume.png")

    numeric = frame.select_dtypes(include="number").drop(columns=[TARGET_COLUMN], errors="ignore")
    if not numeric.empty:
        correlations = numeric.corr(numeric_only=True)
        figure, axis = plt.subplots(figsize=(9, 7))
        image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=70, ha="right")
        axis.set_yticks(range(len(correlations.columns)), correlations.columns)
        axis.set_title("Feature correlation (numeric columns only)")
        figure.colorbar(image, ax=axis, shrink=0.8)
        _save_figure(figure, "numeric_correlation.png")

    missing = frame.drop(columns=["Fraud_Status"], errors="ignore").isna().sum()
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(missing.index.astype(str), missing.values, color="#22c55e")
    axis.set(title="Missing values by column", ylabel="Missing cells")
    axis.tick_params(axis="x", rotation=70)
    _save_figure(figure, "missing_values.png")

    insights: dict[str, Any] = {
        "rows": int(len(frame)),
        "fraud_labeled_transactions": int(frame[TARGET_COLUMN].sum()),
        "legitimate_labeled_transactions": int((frame[TARGET_COLUMN] == 0).sum()),
        "fraud_rate": float(frame[TARGET_COLUMN].mean()),
        "total_transaction_value": float(frame["Transaction_Amount"].sum()),
        "mean_transaction_amount": float(frame["Transaction_Amount"].mean()),
        "daily_transaction_volume_peak": int(daily["size"].max()) if not daily.empty else 0,
        "fraud_rate_by_transaction_type": {
            str(key): float(value)
            for key, value in frame.groupby("Transaction_Type", observed=True)[
                TARGET_COLUMN
            ].mean().items()
        }
        if "Transaction_Type" in frame.columns
        else {},
        "missing_values_by_column": {
            str(key): int(value) for key, value in missing.items() if value
        },
        "interpretation_note": (
            "These are associations in the supplied synthetic dataset; they do not "
            "establish causation or real-world fraud performance."
        ),
    }
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSIGHTS_PATH.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    return insights


if __name__ == "__main__":
    print(json.dumps(run_eda(), indent=2))
