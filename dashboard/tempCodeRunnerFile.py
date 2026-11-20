"""Interactive Streamlit dashboard for the supplied synthetic fraud dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import TARGET_COLUMN
from src.predict import load_model_bundle, score_transactions
from src.train_model import METRICS_PATH, PREDICTIONS_PATH

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transactions_cleaned.csv"
NAVY = "#0b1220"
PANEL = "#121c2f"
BLUE = "#38bdf8"
PURPLE = "#a78bfa"
GREEN = "#34d399"
RED = "#fb7185"
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [BLUE, PURPLE, GREEN, RED]

st.set_page_config(
    page_title="Financial Fraud Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background: {NAVY}; color: #e5e7eb; }}
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {{ background: {NAVY}; }}
    [data-testid="stSidebar"] {{ background: #101a2b; }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    .stMarkdown p,
    .stMarkdown li {{ color: #cbd5e1 !important; }}
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid #26354d; padding: 16px;
        border-radius: 12px;
    }}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {{ color: #e2e8f0 !important; }}
    [data-testid="stMetricLabel"] * {{ color: #94a3b8 !important; }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {{
        color: #f8fafc !important;
        font-size: 1.25rem !important;
    }}
    h1, h2, h3 {{ color: #f8fafc; }}
    .caption-box {{
        background: #17243a; border-left: 4px solid {BLUE}; padding: 12px 16px;
        border-radius: 6px; color: #dbeafe;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_transactions(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce")
    return dataframe


@st.cache_data(show_spinner="Scoring transactions for review...")
def load_scored_transactions(path: str, model_path: str) -> tuple[pd.DataFrame, dict]:
    dataframe = load_transactions(path)
    bundle = load_model_bundle(Path(model_path))
    return score_transactions(dataframe, bundle), bundle


@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict:
    metrics_path = Path(path)
    if not metrics_path.is_file():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def money(value: float) -> str:
    return f"{value:,.2f}"


def show_sidebar(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    st.sidebar.title("Fraud Monitor")
    page = st.sidebar.radio(
        "Dashboard section",
        [
            "Executive Overview",
            "Transaction Analytics",
            "Fraud Detection",
            "Model Performance",
            "Business Insights",
        ],
    )
    st.sidebar.caption("Filters apply to transaction and overview sections.")

    start_date = frame["Date"].min().date()
    end_date = frame["Date"].max().date()
    chosen_dates = st.sidebar.date_input(
        "Transaction dates",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date,
    )
    if isinstance(chosen_dates, tuple) and len(chosen_dates) == 2:
        start, end = chosen_dates
    else:
        start, end = start_date, end_date

    amount_min = float(frame["Transaction_Amount"].min())
    amount_max = float(frame["Transaction_Amount"].max())
    amount_range = st.sidebar.slider(
        "Transaction amount (dataset units)",
        min_value=amount_min,
        max_value=amount_max,
        value=(amount_min, amount_max),
    )

    filtered = frame.loc[
        frame["Date"].dt.date.between(start, end)
        & frame["Transaction_Amount"].between(amount_range[0], amount_range[1])
    ].copy()
    for column, label in [
        ("Transaction_Type", "Transaction type"),
        ("Merchant_Category", "Merchant category"),
    ]:
        if column in frame.columns:
            choices = sorted(frame[column].dropna().astype(str).unique())
            selected = st.sidebar.multiselect(label, choices, default=choices)
            filtered = filtered.loc[filtered[column].astype(str).isin(selected)]
    return filtered, page


def render_overview(frame: pd.DataFrame, bundle: dict, metrics: dict) -> None:
    st.title("Financial Fraud Detection")
    st.caption("Executive Overview | Filtered transaction view")
    labeled_fraud = int(frame[TARGET_COLUMN].eq(1).sum())
    predicted_flags = int(frame["Model_Flag"].sum())
    fraud_rate = float(frame[TARGET_COLUMN].mean()) if len(frame) else 0.0
    flagged_value = float(
        frame.loc[frame["Model_Flag"], "Transaction_Amount"].sum()
    )

    kpis = [
        ("Transactions", f"{len(frame):,}"),
        ("Total value", money(frame["Transaction_Amount"].sum())),
        ("Fraud labels", f"{labeled_fraud:,}"),
        ("Legit labels", f"{int(frame[TARGET_COLUMN].eq(0).sum()):,}"),
        ("Fraud rate", f"{fraud_rate:.1%}"),
        ("Flagged value", money(flagged_value)),
    ]
    for start in range(0, len(kpis), 2):
        columns = st.columns(2)
        for column, (label, value) in zip(columns, kpis[start : start + 2]):
            column.metric(label, value)

    if not bundle["model_qualified_for_real_world_use"]:
        st.warning(
            "Prototype limitation: the current model did not meet the project's "
            "exploratory validation gate. Scores and flags are for review/demo only, "
            "not a reliable fraud decision."
        )
    st.markdown(
        f'<div class="caption-box">Model flags: <b>{predicted_flags:,}</b> rows at '
        f'probability >= {bundle["thresholds"]["decision_threshold"]:.0%}. '
        "A model flag is a review signal; it is not the same as a recorded fraud label."
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    daily = (
        frame.groupby(frame["Date"].dt.date, observed=True)
        .agg(Transactions=("Transaction_ID", "count"), Amount=("Transaction_Amount", "sum"))
        .reset_index(names="Date")
    )
    left.plotly_chart(
        px.line(daily, x="Date", y="Transactions", title="Daily transaction volume", color_discrete_sequence=[BLUE]),
        width="stretch",
    )
    category = (
        frame.groupby("Merchant_Category", observed=True)[TARGET_COLUMN]
        .mean()
        .mul(100)
        .rename("Recorded fraud-label rate (%)")
        .reset_index()
        .sort_values("Recorded fraud-label rate (%)", ascending=False)
    )
    right.plotly_chart(
        px.bar(category, x="Merchant_Category", y="Recorded fraud-label rate (%)", title="Recorded label rate by category", color_discrete_sequence=[PURPLE]),
        width="stretch",
    )

    st.subheader("Latest model-flagged transactions")
    columns = [
        name
        for name in [
            "Transaction_ID",
            "Date",
            "Transaction_Amount",
            "Transaction_Type",
            "Merchant_Category",
            "Fraud_Probability",
            "Risk_Category",
            TARGET_COLUMN,
        ]
        if name in frame.columns
    ]
    flagged = frame.loc[frame["Model_Flag"]].sort_values(
        "Fraud_Probability", ascending=False
    )
    latest = flagged[columns].head(20).copy()
    if "Date" in latest.columns:
        latest["Date"] = pd.to_datetime(latest["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(latest, width="stretch", hide_index=True)
    st.caption(
        f"Selected model: {bundle['model_name']}. "
        "Amounts are shown in dataset units; no currency is specified in the source."
    )


def render_analytics(frame: pd.DataFrame) -> None:
    st.title("Transaction Analytics")
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(
            frame,
            x="Transaction_Amount",
            color=TARGET_COLUMN,
            nbins=50,
            barmode="overlay",
            title="Transaction amount distribution by recorded label",
            color_discrete_map={0: BLUE, 1: RED},
        ),
        width="stretch",
    )
    right.plotly_chart(
        px.box(
            frame,
            x="Transaction_Type",
            y="Transaction_Amount",
            color=TARGET_COLUMN,
            title="Amount by transaction type and label",
            color_discrete_map={0: BLUE, 1: RED},
        ),
        width="stretch",
    )
    by_type = (
        frame.groupby("Transaction_Type", observed=True)
        .agg(
            Transactions=("Transaction_ID", "count"),
            Fraud_Label_Rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    by_type["Fraud_Label_Rate"] *= 100
    st.plotly_chart(
        px.bar(
            by_type,
            x="Transaction_Type",
            y="Transactions",
            color="Fraud_Label_Rate",
            title="Transaction volume by type",
            color_continuous_scale="Purples",
        ),
        width="stretch",
    )
    category = (
        frame.groupby("Merchant_Category", observed=True)
        .agg(
            Transactions=("Transaction_ID", "count"),
            Fraud_Label_Rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    category["Fraud_Label_Rate"] *= 100
    st.dataframe(category.sort_values("Fraud_Label_Rate", ascending=False), width="stretch", hide_index=True)


def render_detection(frame: pd.DataFrame, bundle: dict) -> None:
    st.title("Fraud Detection Review Queue")
    st.caption("Scores are experimental model outputs, not proof of fraud.")
    risk_choices = st.multiselect(
        "Risk bands",
        ["Low", "Medium", "High"],
        default=["Medium", "High"],
    )
    flagged_only = st.checkbox("Show model-flagged rows only", value=True)
    shown = frame.loc[frame["Risk_Category"].isin(risk_choices)].copy()
    if flagged_only:
        shown = shown.loc[shown["Model_Flag"]]
    columns = [
        name
        for name in [
            "Transaction_ID",
            "Date",
            "Transaction_Amount",
            "Transaction_Type",
            "Merchant_Category",
            "Device_Type",
            "Location",
            "Fraud_Probability",
            "Risk_Category",
            "Model_Flag",
            TARGET_COLUMN,
        ]
        if name in shown.columns
    ]
    st.metric("Rows in current review queue", f"{len(shown):,}")
    table = shown.sort_values("Fraud_Probability", ascending=False)[columns].copy()
    if "Date" in table.columns:
        table["Date"] = pd.to_datetime(table["Date"]).dt.strftime("%Y-%m-%d")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Risk bands: Low < 30%, Medium 30% to < 65%, High >= 65%. "
        f"Model flag threshold: >= {bundle['thresholds']['decision_threshold']:.0%}."
    )


def render_performance(metrics: dict) -> None:
    st.title("Model Performance")
    if not metrics:
        st.error("Model metrics are missing. Run `python -m src.train_model` first.")
        return
    st.caption(
        f"Chronological holdout begins {metrics['chronological_test_start']}; "
        f"{metrics['test_rows']:,} later transactions were held out."
    )
    if not metrics["model_qualified_for_real_world_use"]:
        st.warning(
            "No model passed the project's exploratory signal gate. "
            "Do not use these predictions to approve, block, or accuse a customer."
        )
    selected = metrics["selected_metrics"]
    cols = st.columns(4)
    for column, name, key in zip(
        cols,
        ["Precision", "Recall", "F1-score", "ROC-AUC"],
        ["precision", "recall", "f1", "roc_auc"],
    ):
        value = selected.get(key)
        column.metric(name, "N/A" if value is None else f"{value:.3f}")
    st.metric(
        "PR-AUC (average precision)",
        f"{selected['pr_auc']:.3f}",
        help=f"Holdout fraud prevalence baseline: {metrics['test_fraud_rate']:.3f}",
    )

    comparison = pd.DataFrame(metrics["model_comparison"]).T.reset_index(names="Model")
    confusion = np.asarray(selected["confusion_matrix"], dtype=int)
    st.plotly_chart(
        px.imshow(
            confusion,
            x=["Predicted legitimate", "Predicted fraud"],
            y=["Actual legitimate", "Actual fraud"],
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Prediction", "y": "Recorded label", "color": "Rows"},
            title="Selected model confusion matrix",
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.bar(
            comparison,
            x="Model",
            y=["precision", "recall", "f1"],
            barmode="group",
            title="Chronological holdout: precision, recall and F1",
            color_discrete_sequence=[BLUE, PURPLE, GREEN],
        ),
        use_container_width=True,
    )
    st.dataframe(comparison, width="stretch", hide_index=True)

    if PREDICTIONS_PATH.is_file():
        predictions = pd.read_csv(PREDICTIONS_PATH)
        y_true = predictions[TARGET_COLUMN].astype(int)
        y_score = predictions["Fraud_Probability"]
        from sklearn.metrics import precision_recall_curve, roc_curve

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        left, right = st.columns(2)
        left.plotly_chart(
            go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line={"color": PURPLE})).update_layout(
                title="Precision-recall curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                template="plotly_dark",
            ),
            width="stretch",
        )
        right.plotly_chart(
            go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line={"color": BLUE})).add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#64748b"})
            ).update_layout(
                title="ROC curve",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                template="plotly_dark",
            ),
            width="stretch",
        )
    st.info(
        "Accuracy is not enough for fraud detection: a majority-class model can look "
        "accurate while finding no fraud. Compare recall, precision, PR-AUC and false "
        "positives, and validate on new real-world data before deployment."
    )


def render_insights(frame: pd.DataFrame) -> None:
    st.title("Business Insights")
    st.caption(
        "These summaries describe recorded labels in this synthetic sample; "
        "they are associations, not causes or real-world guarantees."
    )
    category = (
        frame.groupby("Merchant_Category", observed=True)
        .agg(
            Transactions=("Transaction_ID", "count"),
            Fraud_Label_Rate=(TARGET_COLUMN, "mean"),
        )
        .query("Transactions >= 100")
        .sort_values("Fraud_Label_Rate", ascending=False)
    )
    if not category.empty:
        top = category.iloc[0]
        st.info(
            f"Highest observed category label rate in this filtered view: "
            f"{category.index[0]} ({top['Fraud_Label_Rate']:.1%}, "
            f"{int(top['Transactions']):,} transactions). This does not imply causation."
        )
        st.dataframe(category.reset_index(), width="stretch", hide_index=True)
    monthly = (
        frame.groupby(frame["Date"].dt.to_period("M"), observed=True)[TARGET_COLUMN]
        .agg(["count", "mean"])
        .reset_index()
    )
    monthly["Date"] = monthly["Date"].astype(str)
    st.plotly_chart(
        px.line(
            monthly,
            x="Date",
            y="mean",
            markers=True,
            title="Monthly recorded fraud-label rate",
            labels={"mean": "Fraud-label rate", "Date": "Month"},
            color_discrete_sequence=[RED],
        ),
        width="stretch",
    )
    st.markdown(
        "- Investigate category and time patterns with domain experts before taking action.\n"
        "- Confirm suspicious transactions using independent evidence and human review.\n"
        "- Obtain representative, verified real transaction labels before training a production model."
    )


def main() -> None:
    if not DATA_PATH.is_file():
        st.error("Processed transactions are missing. Run `python -m src.data_cleaning` first.")
        st.stop()
    try:
        frame, bundle = load_scored_transactions(str(DATA_PATH), str(bundle_path()))
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    if frame.empty:
        st.warning("No transactions are available after loading the processed dataset.")
        st.stop()
    filtered, page = show_sidebar(frame)
    metrics = load_metrics(str(METRICS_PATH))
    if filtered.empty and page != "Model Performance":
        st.warning("No transactions match the selected filters. Adjust the sidebar filters.")
        return
    if page == "Executive Overview":
        render_overview(filtered, bundle, metrics)
    elif page == "Transaction Analytics":
        render_analytics(filtered)
    elif page == "Fraud Detection":
        render_detection(filtered, bundle)
    elif page == "Model Performance":
        render_performance(metrics)
    else:
        render_insights(filtered)


def bundle_path() -> Path:
    return PROJECT_ROOT / "models" / "fraud_detection_model.joblib"


if __name__ == "__main__":
    main()
