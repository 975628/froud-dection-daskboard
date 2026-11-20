# Financial Fraud Detection: Project Report

## Executive summary

This project analyzes the supplied synthetic transaction CSV and builds a
leakage-aware demonstration workflow: data validation, exploratory charts,
chronological model comparison, and a Streamlit review dashboard. The source
contains a `Fraud_Label` column, so a supervised comparison is possible.
However, the tested predictors did **not** show useful out-of-time predictive
signal. The model is not suitable for real financial decisions.

## Dataset and scope

- Source: `data/raw/synthetic_fraud_dataset1.csv` (copied byte-for-byte; raw
  source remains unchanged).
- Size: 50,000 transactions, 14 columns.
- Target: `Fraud_Label`; 0 means the supplied row is labeled legitimate, 1
  means it is labeled fraud. These are dataset labels, not independently
  verified real-world outcomes.
- Distribution: 33,933 label 0 (67.866%) and 16,067 label 1 (32.134%); the
  majority/minority ratio is approximately 2.11:1.
- Dates: every row parsed; 1 January through 31 December 2023. Only a day is
  available, not an hour or minute.
- Quality checks: 0 missing cells, 0 exact duplicate rows, 0 negative
  transaction amounts, 0 negative card ages, and no invalid target labels.
- Amount units/currency are not defined by the source.

## Methodology

1. Keep the provided CSV in `data/raw/`; write a separate cleaned CSV to
   `data/processed/transactions_cleaned.csv`.
2. Normalize text and parse dates, validate the binary target, and record a
   JSON cleaning audit. Missing predictor values (if any) are imputed inside
   the model pipeline so medians/categories are learned from training rows
   only.
3. Use a chronological holdout beginning 1 October 2023: 37,462 earlier
   transactions for training and 12,538 later transactions for testing.
4. Exclude `Fraud_Label`, `Transaction_ID`, and `User_ID` from predictors.
   Derive calendar month and weekday from `Date`. Scale numeric inputs and
   one-hot encode categories inside each pipeline.
5. Compare Logistic Regression, Decision Tree, and Random Forest using the
   same holdout and probability threshold 0.50. Select the candidate with
   highest recall, then F1, then PR-AUC. This selection is for demonstration,
   not production deployment.

## Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.322 | 0.337 | 0.329 | 0.497 | 0.319 | 0.558 |
| Decision Tree (selected by recall) | 0.319 | 0.432 | 0.367 | 0.497 | 0.318 | 0.520 |
| Random Forest | 0.314 | 0.103 | 0.155 | 0.495 | 0.317 | 0.638 |

The holdout fraud prevalence is 0.322, which is the no-ranking baseline for
average precision. The Decision Tree's PR-AUC (0.318) is slightly below that
baseline and its ROC-AUC (0.497) is effectively chance. A majority-class model
has 0.678 accuracy while detecting no fraud, illustrating why accuracy alone is
misleading here. At the selected model's 0.50 threshold, the confusion matrix
is `[[4774, 3724], [2294, 1746]]` (rows: actual 0/1; columns: predicted 0/1).
The recall trade-off comes with many false positives.

The project's exploratory qualification gate is ROC-AUC >= 0.55 and PR-AUC at
least 0.03 above holdout prevalence. No candidate passed. These cutoffs are
transparent teaching/demo gates, not external standards or proof of safety.

## EDA observations

- The recorded fraud-label rate is about 32.1%, and is stable between the
  training time period (32.10%) and holdout period (32.22%).
- Rows with `Previous_Fraudulent_Activity` set to 1 have a 32.0% fraud-label
  rate; rows set to 0 have a 32.1% rate in this sample.
- Transaction type label rates are close: ATM Withdrawal 32.49%, Bank Transfer
  32.08%, Online 32.13%, POS 31.84%.
- The four sequential date quarters also show similar label rates. This and
  the near-chance model results suggest that the provided features contain
  little demonstrated information about the synthetic labels.
- `reports/figures/` contains class distribution, amount distribution,
  transaction-type rate, daily volume, numeric correlation, and missing-value
  charts. The category differences are descriptive associations only; they do
  not establish causation.

## Dashboard and risk communication

The Streamlit dashboard has five sections: Executive Overview, Transaction
Analytics, Fraud Detection, Model Performance, and Business Insights. Filters
cover date, amount, transaction type, and merchant category. Displayed risk
bands are Low (<30%), Medium (30% to <65%), and High (>=65%); the model-flag
threshold is 50%. The dashboard separates model flags from recorded dataset
labels and clearly warns that model scores are experimental. It omits `User_ID`
from transaction tables.

## Limitations and recommendations

- The dataset is synthetic; verify how its labels were generated before
  interpreting them as ground truth.
- The source only has day-level dates and no currency.
- Near-chance holdout metrics mean the saved candidate must not be used to
  block/approve transactions or accuse customers.
- Obtain verified, representative, appropriately governed real transaction
  data; investigate target construction and add meaningful pre-transaction
  signals.
- Before any deployment, validate on future and external populations, review
  false-positive costs with stakeholders, calibrate thresholds, monitor drift,
  and require human investigation.

## Reproducibility

Run from the project root:

```powershell
python -m src.data_cleaning
python -m src.eda
python -m src.train_model
pytest
streamlit run dashboard/app.py
```

Outputs include a cleaned CSV, cleaning/profile JSON, EDA figures, model
comparison and holdout predictions, model metrics JSON, and a Joblib model
bundle.
