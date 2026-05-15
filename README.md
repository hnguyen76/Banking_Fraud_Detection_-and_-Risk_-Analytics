# Banking Fraud Detection & Risk Analytics

Professional fraud analytics project with a Streamlit dashboard, exploratory analysis, supervised fraud classification, anomaly detection, risk scoring, predictive scoring, and financial exposure analytics.

Created by Hieu Nguyen

## Project Highlights

- Exploratory Data Analysis (EDA) across transaction amount, channel, authentication type, time of day, and fraud behavior.
- Fraud Detection Models using Logistic Regression and Random Forest classifiers with stratified holdout validation.
- Risk Scoring Systems with a weighted risk score, tiering logic, review queue, and expected exposure estimate.
- Predictive Analytics form for scoring a new banking transaction.
- Dashboard Development with interactive filters, KPI cards, Plotly charts, model metrics, and high-risk transaction tables.
- Classification & Anomaly Detection using supervised classification and Isolation Forest anomaly scoring.
- Financial Analytics Projects covering channel exposure, amount bands, fraud rates, and expected loss views.

## Quick Start

```powershell
python -m pip install -r requirements.txt
streamlit run dashboard.py
```

Open the local Streamlit URL in your browser. The dashboard reads `banking_transactions.csv` from the project root.

## Notebook Environment

```powershell
python -m venv .bankingvenv
.\.bankingvenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name banking-risk --display-name "Banking Risk Analytics"
```

Then open `notebooks/Banking_Fraud_Risk_Analytics_Workflow.ipynb` and select the `Banking Risk Analytics` kernel.

## Repository Structure

```text
.
|-- banking_transactions.csv
|-- dashboard.py
|-- notebooks/
|   `-- Banking_Fraud_Risk_Analytics_Workflow.ipynb
|-- reports/
|   `-- Banking_Fraud_Risk_Analytics_Report.md
|-- requirements.txt
|-- .streamlit/
|   `-- config.toml
`-- README.md
```

## Dataset Snapshot

- 10,000 transactions
- 20 original columns
- 12.51% confirmed fraud rate
- No missing values in the provided CSV
- Core signals include anomaly score, transaction amount, device risk, velocity, failed transactions, geography, authentication, and payment channel.

## Model Benchmark

The included benchmark uses a 75/25 stratified holdout split.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9783 | 0.8678 | 0.8018 | 0.8403 | 0.8206 |
| Random Forest | 0.9722 | 0.8121 | 0.7745 | 0.8339 | 0.8031 |
| Isolation Forest | 0.7110 | 0.2935 | n/a | n/a | n/a |

## Author

Created by Hieu Nguyen
