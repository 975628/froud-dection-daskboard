"""Train leakage-aware baseline models and record their holdout metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_cleaning import (
    PROJECT_ROOT,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    clean_transactions,
    load_dataset,
)

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "model_evaluation_predictions.csv"
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5


def prepare_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create prediction-time features and remove identifiers/target leakage."""
    features = dataframe.drop(
        columns=[TARGET_COLUMN, "Transaction_ID", "User_ID", "Date"],
        errors="ignore",
    ).copy()
    if "Date" in dataframe.columns:
        dates = pd.to_datetime(dataframe["Date"], errors="coerce")
        features["Transaction_Month"] = dates.dt.month
        features["Transaction_Weekday"] = dates.dt.dayofweek
    return features


def chronological_split(
    dataframe: pd.DataFrame,
    test_fraction: float = 0.25,
) -> tuple[pd.Index, pd.Index, str]:
    """Keep later dates in the holdout set when enough dates are available."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if "Date" in dataframe.columns:
        dates = pd.to_datetime(dataframe["Date"], errors="coerce")
        unique_dates = np.sort(dates.dropna().unique())
        if len(unique_dates) >= 4 and dates.notna().all():
            cutoff_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * (1 - test_fraction))))
            cutoff = pd.Timestamp(unique_dates[cutoff_index])
            train_index = dataframe.index[dates < cutoff]
            test_index = dataframe.index[dates >= cutoff]
            if len(train_index) and len(test_index):
                return train_index, test_index, cutoff.date().isoformat()
    raise ValueError(
        "A chronological split needs a valid Date value on every row and at least "
        "four distinct dates. Review the dataset instead of silently using a random split."
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Fit imputers, scaling, and category encoding only on training data."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.columns.difference(numeric_columns).tolist()
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No usable predictor features remain after excluding identifiers.")
    return ColumnTransformer(transformers, remainder="drop")


def build_models() -> dict[str, Any]:
    """Return a small, explainable comparison set for this tabular dataset."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def _measure(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray]:
    predicted = model.predict(X_test).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_test, probabilities))
            if y_test.nunique() == 2
            else None
        ),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
    }
    return metrics, probabilities


def train_and_evaluate() -> dict[str, Any]:
    """Compare candidates on a chronological holdout and save reproducible outputs."""
    if PROCESSED_DATA_PATH.exists():
        dataframe = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        dataframe, _ = clean_transactions(load_dataset())

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing.")
    dataframe[TARGET_COLUMN] = pd.to_numeric(dataframe[TARGET_COLUMN], errors="raise").astype(int)
    if not set(dataframe[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1.")

    train_index, test_index, cutoff = chronological_split(dataframe)
    features = prepare_features(dataframe)
    target = dataframe[TARGET_COLUMN]
    X_train, X_test = features.loc[train_index], features.loc[test_index]
    y_train, y_test = target.loc[train_index], target.loc[test_index]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both chronological partitions must contain both target classes.")

    comparison: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}
    holdout_probabilities: dict[str, np.ndarray] = {}
    for name, estimator in build_models().items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics, probabilities = _measure(pipeline, X_test, y_test)
        comparison[name] = metrics
        fitted_models[name] = pipeline
        holdout_probabilities[name] = probabilities

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)
    majority_accuracy = float(accuracy_score(y_test, dummy_predictions))
    baseline_pr_auc = float(y_test.mean())

    selected_name = max(
        comparison,
        key=lambda name: (
            comparison[name]["recall"],
            comparison[name]["f1"],
            comparison[name]["pr_auc"],
        ),
    )
    selected_metrics = comparison[selected_name]
    # These are explicit project screening gates, not regulatory standards.
    model_qualified = bool(
        selected_metrics["roc_auc"] is not None
        and selected_metrics["roc_auc"] >= 0.55
        and selected_metrics["pr_auc"] >= baseline_pr_auc + 0.03
    )
    thresholds = {
        "decision_threshold": DECISION_THRESHOLD,
        "high_risk_probability": 0.65,
        "medium_risk_probability": 0.30,
    }

    evaluation_predictions = dataframe.loc[
        test_index, ["Transaction_ID", "Date", "Transaction_Amount", TARGET_COLUMN]
    ].copy()
    evaluation_predictions["Fraud_Probability"] = holdout_probabilities[selected_name]
    evaluation_predictions["Predicted_Fraud"] = (
        evaluation_predictions["Fraud_Probability"] >= DECISION_THRESHOLD
    ).astype(int)

    metrics_document = {
        "selected_model": selected_name,
        "selection_rule": "Highest holdout recall, then F1, then PR-AUC at probability threshold 0.50.",
        "model_qualified_for_real_world_use": model_qualified,
        "qualification_rule": (
            "Exploratory gate: ROC-AUC >= 0.55 and PR-AUC >= holdout fraud prevalence + 0.03. "
            "Passing this gate would not replace external validation."
        ),
        "dataset_rows": int(len(dataframe)),
        "training_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "chronological_test_start": cutoff,
        "training_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": baseline_pr_auc,
        "majority_class_baseline_accuracy": majority_accuracy,
        "decision_thresholds": thresholds,
        "selected_metrics": selected_metrics,
        "model_comparison": comparison,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": fitted_models[selected_name],
        "model_name": selected_name,
        "feature_columns": features.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "thresholds": thresholds,
        "model_qualified_for_real_world_use": model_qualified,
    }
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics_document, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).T.rename_axis("model").to_csv(COMPARISON_PATH)
    evaluation_predictions.to_csv(PREDICTIONS_PATH, index=False)
    return metrics_document


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
