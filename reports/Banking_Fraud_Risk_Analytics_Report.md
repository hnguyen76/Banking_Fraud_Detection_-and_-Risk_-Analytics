# Banking Fraud Detection & Risk Analytics Report

Created by Hieu Nguyen

Report date: May 15, 2026

## Executive Summary

This project delivers a professional fraud risk analytics workflow for banking transactions. It includes Exploratory Data Analysis (EDA), Fraud Detection Models, Risk Scoring Systems, Predictive Analytics, Dashboard Development, Classification & Anomaly Detection, and Financial Analytics Projects.

The dataset contains 10,000 transactions with 20 original fields and no missing values. Confirmed fraud appears in 1,251 transactions, creating a fraud rate of 12.51%. Total monitored transaction volume is $124.13M, with $15.66M tied to confirmed fraudulent transactions.

The strongest predictive signal is `anomaly_score`. Confirmed fraud transactions have an average anomaly score of 0.772, compared with 0.288 for legitimate transactions. Transactions with `anomaly_score >= 0.75` show an observed fraud rate of 88.80%, making this feature a high-priority signal for model monitoring and rules-based review queues.

## 1. Exploratory Data Analysis (EDA)

### Dataset Profile

| Metric | Value |
| --- | ---: |
| Transactions | 10,000 |
| Original columns | 20 |
| Missing values | 0 |
| Fraud transactions | 1,251 |
| Legitimate transactions | 8,749 |
| Fraud rate | 12.51% |
| Total transaction volume | $124.13M |
| Confirmed fraud value | $15.66M |

### Channel Distribution

| Payment Channel | Transactions | Fraud Rate |
| --- | ---: | ---: |
| Mobile App | 4,804 | 12.82% |
| Web Banking | 3,247 | 12.32% |
| POS Terminal | 1,172 | 11.43% |
| ATM | 777 | 13.00% |

### Authentication Distribution

| Authentication Type | Transactions | Fraud Rate |
| --- | ---: | ---: |
| OTP | 4,214 | 13.08% |
| Two-Factor Authentication | 2,384 | 11.62% |
| Password Only | 1,817 | 12.77% |
| Biometric | 1,585 | 12.05% |

### Key EDA Findings

- Fraud is materially more concentrated at higher anomaly scores than at higher transaction amounts alone.
- Average fraudulent transaction amount is $12,515.91, close to the legitimate average of $12,398.00, so amount alone is not a sufficient fraud rule.
- The highest hourly fraud rates appear at hours 4, 18, 9, 22, and 13.
- ATM has the highest channel-level fraud rate at 13.00%, but Mobile App carries the largest transaction count and therefore deserves focused exposure monitoring.

## 2. Fraud Detection Models

The supervised modeling workflow trains two classification models:

- Logistic Regression with class balancing for a strong interpretable baseline.
- Random Forest with class balancing for non-linear interaction capture.

Validation uses a 75/25 stratified holdout split to preserve the fraud class distribution.

| Model | ROC-AUC | PR-AUC | Best Threshold | Precision | Recall | F1 | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.9783 | 0.8678 | 0.8225 | 0.8018 | 0.8403 | 0.8206 | TN 2122, FP 65, FN 50, TP 263 |
| Random Forest | 0.9722 | 0.8121 | 0.7176 | 0.7745 | 0.8339 | 0.8031 | TN 2111, FP 76, FN 52, TP 261 |

Logistic Regression is the benchmark champion based on PR-AUC and F1. PR-AUC is emphasized because fraud detection is an imbalanced classification problem where precision-recall behavior is more operationally relevant than accuracy.

## 3. Risk Scoring Systems

The dashboard includes a weighted operational risk score from 0 to 100. The score combines model-relevant indicators and rules-based risk features:

| Component | Weight |
| --- | ---: |
| Anomaly score | 36% |
| Device risk score | 12% |
| Transaction velocity score | 12% |
| Failed transactions last 30 days | 10% |
| Login attempts | 8% |
| Geographic distance | 7% |
| Transaction amount | 6% |
| Suspicious IP flag | 4% |
| International transaction flag | 3% |
| Card-not-present adjustment | 2% |

Risk tiers:

| Tier | Score Range | Action |
| --- | ---: | --- |
| Low | 0-35 | Standard monitoring |
| Watch | 35-55 | Monitor for pattern changes |
| High | 55-75 | Enhanced review |
| Critical | 75-100 | Immediate fraud operations review |

## 4. Predictive Analytics

The predictive scoring module lets an analyst enter a new transaction scenario and receive:

- Rule-based risk score.
- Risk tier.
- Champion classifier fraud probability.

This supports operational triage before a transaction is escalated to manual review, secondary authentication, or case management.

## 5. Dashboard Development

The Streamlit dashboard is designed for a professional fraud analytics workflow:

- Executive KPI strip for transaction count, total volume, fraud rate, confirmed fraud value, and critical reviews.
- Sidebar filters for payment channel, authentication type, risk tier, transaction hour, and transaction amount.
- EDA views for distribution analysis, channel behavior, authentication behavior, anomaly score behavior, and correlation review.
- Model tab with classification metrics, confusion matrix, feature importance, and anomaly detection metrics.
- Risk scoring tab with threshold controls, review queue, expected exposure, and high-risk transaction table.
- Predictive analytics tab for scoring new transaction scenarios.
- Financial analytics tab for channel exposure, amount bands, and expected loss breakdowns.

## 6. Classification & Anomaly Detection

The project includes two complementary fraud detection approaches:

- Classification identifies known fraud patterns using labeled outcomes.
- Isolation Forest anomaly detection identifies unusual transaction behavior without depending on labels.

Isolation Forest benchmark:

| Method | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Isolation Forest | 0.7110 | 0.2935 |

The anomaly detector is useful as a secondary control and investigation trigger, while the supervised classifier is better suited for primary fraud prediction on this labeled dataset.

## 7. Financial Analytics Projects

The financial analytics layer focuses on business impact:

- Confirmed fraud exposure by transaction channel.
- Expected loss by risk tier.
- Fraud rate by transaction amount band.
- Review queue value based on risk threshold.
- Prioritized transaction table for operations teams.

This turns technical fraud detection into an action-oriented portfolio monitoring tool.

## Recommendations

1. Use Logistic Regression as the current champion model because it provides the strongest PR-AUC and stable recall on the holdout set.
2. Treat high anomaly score as the primary monitoring signal, especially when combined with high velocity, suspicious IP, or international transaction flags.
3. Review Critical tier transactions first, then High tier transactions with elevated transaction value.
4. Monitor Mobile App exposure closely because it has the largest transaction volume.
5. Retrain the model regularly as fraud patterns change, and track PR-AUC, recall, false positives, and manual-review capacity over time.

## Limitations

- The dataset does not include event dates, customer identifiers, merchant identifiers, chargeback recovery, or confirmed loss amount.
- The current risk score estimates exposure using transaction amount and risk intensity, not actual recovered or unrecovered financial loss.
- Model validation uses a single stratified holdout split; production deployment should add time-based validation, drift monitoring, and threshold governance.

Created by Hieu Nguyen
