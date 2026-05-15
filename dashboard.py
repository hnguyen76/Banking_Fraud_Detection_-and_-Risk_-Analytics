from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).with_name("banking_transactions.csv")
CREATOR = "Hieu Nguyen"
RANDOM_STATE = 42


st.set_page_config(
    page_title="Banking Fraud Detection & Risk Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }
    .app-header {
        border: 1px solid #dbe4ee;
        background: linear-gradient(135deg, #ffffff 0%, #eef7f5 45%, #f8fbff 100%);
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
    }
    .app-title {
        color: #102a43;
        font-size: 2.1rem;
        font-weight: 760;
        line-height: 1.1;
        letter-spacing: 0;
        margin: 0;
    }
    .app-subtitle {
        color: #486581;
        font-size: 1rem;
        margin-top: 0.35rem;
        max-width: 980px;
    }
    .creator {
        display: inline-block;
        margin-top: 0.85rem;
        color: #0f766e;
        font-weight: 700;
        border: 1px solid #99f6e4;
        background: #ecfdf5;
        border-radius: 999px;
        padding: 0.28rem 0.72rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 8px 22px rgba(17, 24, 39, 0.045);
    }
    div[data-testid="stMetricLabel"] p {
        color: #486581;
        font-weight: 650;
    }
    div[data-testid="stMetricValue"] {
        color: #102a43;
    }
    .section-note {
        border-left: 4px solid #0f766e;
        background: #f8fbff;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        color: #334e68;
        margin-bottom: 0.9rem;
    }
    .small-caption {
        color: #627d98;
        font-size: 0.86rem;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #dbe4ee;
        color: #486581;
        text-align: center;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass(frozen=True)
class ModelMetrics:
    model: str
    roc_auc: float
    pr_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value * 100:,.2f}%"


@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fraud_flag"] = df["fraud_flag"].astype(bool)
    df["fraud_label"] = np.where(df["fraud_flag"], "Fraud", "Legitimate")
    return add_risk_features(df)


def normalize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo = float(series.quantile(lower_q))
    hi = float(series.quantile(upper_q))
    clipped = series.clip(lo, hi)
    span = hi - lo
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (clipped - lo) / span


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    risk_components = {
        "anomaly_score": 0.36,
        "device_risk_score": 0.12,
        "transaction_velocity_score": 0.12,
        "failed_transactions_last_30d": 0.10,
        "login_attempts": 0.08,
        "geo_distance_km": 0.07,
        "transaction_amount": 0.06,
        "suspicious_ip_flag": 0.04,
        "international_transaction_flag": 0.03,
    }
    weighted_score = pd.Series(np.zeros(len(result)), index=result.index, dtype=float)
    for column, weight in risk_components.items():
        if column in {"anomaly_score", "suspicious_ip_flag", "international_transaction_flag"}:
            component = result[column].astype(float)
        else:
            component = normalize(result[column].astype(float))
        weighted_score += component * weight

    card_not_present = 1 - result["card_present_flag"].astype(float)
    weighted_score += card_not_present * 0.02

    result["risk_score"] = (weighted_score * 100).clip(0, 100).round(1)
    result["risk_tier"] = pd.cut(
        result["risk_score"],
        bins=[-0.1, 35, 55, 75, 100.1],
        labels=["Low", "Watch", "High", "Critical"],
    ).astype(str)
    result["expected_loss"] = (result["transaction_amount"] * result["risk_score"] / 100).round(2)
    result["rules_alert"] = np.where(
        (result["risk_score"] >= 75)
        | (result["anomaly_score"] >= 0.75)
        | ((result["suspicious_ip_flag"] == 1) & (result["international_transaction_flag"] == 1)),
        "Review",
        "Monitor",
    )
    return result


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Portfolio Filters")

    channels = sorted(df["payment_channel"].unique())
    selected_channels = st.sidebar.multiselect("Payment channel", channels, default=channels)

    auth_types = sorted(df["authentication_type"].unique())
    selected_auth = st.sidebar.multiselect("Authentication type", auth_types, default=auth_types)

    tiers = ["Low", "Watch", "High", "Critical"]
    selected_tiers = st.sidebar.multiselect("Risk tier", tiers, default=tiers)

    hour_range = st.sidebar.slider(
        "Transaction hour",
        min_value=int(df["transaction_time_hour"].min()),
        max_value=int(df["transaction_time_hour"].max()),
        value=(int(df["transaction_time_hour"].min()), int(df["transaction_time_hour"].max())),
    )

    amount_range = st.sidebar.slider(
        "Transaction amount",
        min_value=float(df["transaction_amount"].min()),
        max_value=float(df["transaction_amount"].max()),
        value=(float(df["transaction_amount"].min()), float(df["transaction_amount"].max())),
        step=100.0,
        format="$%.0f",
    )

    filtered = df[
        df["payment_channel"].isin(selected_channels)
        & df["authentication_type"].isin(selected_auth)
        & df["risk_tier"].isin(selected_tiers)
        & df["transaction_time_hour"].between(hour_range[0], hour_range[1])
        & df["transaction_amount"].between(amount_range[0], amount_range[1])
    ]
    return filtered


def render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <p class="app-title">Banking Fraud Detection & Risk Analytics</p>
            <div class="app-subtitle">
                Executive-grade fraud intelligence dashboard covering EDA, classification models,
                anomaly detection, predictive scoring, risk tiers, and financial exposure analytics.
            </div>
            <span class="creator">Created by {CREATOR}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame) -> None:
    fraud = df["fraud_flag"]
    fraud_rate = float(fraud.mean()) if len(df) else 0.0
    fraud_amount = float(df.loc[fraud, "transaction_amount"].sum()) if len(df) else 0.0
    expected_loss = float(df["expected_loss"].sum()) if len(df) else 0.0
    critical_count = int((df["risk_tier"] == "Critical").sum()) if len(df) else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Total Volume", money(float(df["transaction_amount"].sum())) if len(df) else "$0")
    c3.metric("Fraud Rate", pct(fraud_rate))
    c4.metric("Confirmed Fraud Value", money(fraud_amount))
    c5.metric("Critical Reviews", f"{critical_count:,}", delta=money(expected_loss), delta_color="off")


def fraud_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, observed=False)
        .agg(
            fraud_rate=("fraud_flag", "mean"),
            transactions=("transaction_id", "count"),
            amount=("transaction_amount", "sum"),
            expected_loss=("expected_loss", "sum"),
        )
        .reset_index()
    )
    grouped["fraud_rate_pct"] = grouped["fraud_rate"] * 100
    return grouped


def render_overview(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <div class="section-note">
            The monitored portfolio is imbalanced, with fraud concentrated around elevated anomaly scores.
            Use the risk tier and channel views to prioritize reviews where transaction value and fraud likelihood overlap.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.25, 1])

    hourly = (
        df.groupby("transaction_time_hour", observed=False)
        .agg(transactions=("transaction_id", "count"), fraud_rate=("fraud_flag", "mean"))
        .reset_index()
    )
    hourly["fraud_rate_pct"] = hourly["fraud_rate"] * 100
    fig_hour = go.Figure()
    fig_hour.add_bar(
        x=hourly["transaction_time_hour"],
        y=hourly["transactions"],
        name="Transactions",
        marker_color="#2563eb",
    )
    fig_hour.add_scatter(
        x=hourly["transaction_time_hour"],
        y=hourly["fraud_rate_pct"],
        name="Fraud rate",
        mode="lines+markers",
        yaxis="y2",
        marker_color="#dc2626",
    )
    fig_hour.update_layout(
        title="Hourly Transaction Load and Fraud Rate",
        xaxis_title="Hour of day",
        yaxis_title="Transactions",
        yaxis2=dict(title="Fraud rate (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        height=420,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    left.plotly_chart(fig_hour, use_container_width=True)

    tier_counts = df["risk_tier"].value_counts().reindex(["Low", "Watch", "High", "Critical"]).fillna(0)
    fig_tier = px.pie(
        names=tier_counts.index,
        values=tier_counts.values,
        hole=0.58,
        title="Risk Tier Distribution",
        color=tier_counts.index,
        color_discrete_map={
            "Low": "#16a34a",
            "Watch": "#ca8a04",
            "High": "#ea580c",
            "Critical": "#dc2626",
        },
    )
    fig_tier.update_traces(textinfo="percent+label")
    fig_tier.update_layout(height=420, margin=dict(l=20, r=20, t=70, b=20))
    right.plotly_chart(fig_tier, use_container_width=True)

    channel = fraud_rate_by(df, "payment_channel").sort_values("expected_loss", ascending=False)
    fig_channel = px.bar(
        channel,
        x="payment_channel",
        y="expected_loss",
        color="fraud_rate_pct",
        text=channel["fraud_rate_pct"].map(lambda x: f"{x:.1f}%"),
        color_continuous_scale=["#0891b2", "#f59e0b", "#dc2626"],
        title="Expected Exposure by Payment Channel",
        labels={
            "payment_channel": "Payment channel",
            "expected_loss": "Expected exposure",
            "fraud_rate_pct": "Fraud rate (%)",
        },
    )
    fig_channel.update_layout(height=420, margin=dict(l=20, r=20, t=70, b=20))
    st.plotly_chart(fig_channel, use_container_width=True)


def render_eda(df: pd.DataFrame) -> None:
    left, right = st.columns(2)

    fig_amount = px.histogram(
        df,
        x="transaction_amount",
        color="fraud_label",
        nbins=45,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Fraud": "#dc2626", "Legitimate": "#2563eb"},
        title="Transaction Amount Distribution",
        labels={"transaction_amount": "Transaction amount", "fraud_label": "Class"},
    )
    fig_amount.update_layout(height=410, margin=dict(l=20, r=20, t=70, b=20))
    left.plotly_chart(fig_amount, use_container_width=True)

    sample = df.sample(min(len(df), 3500), random_state=RANDOM_STATE) if len(df) else df
    fig_scatter = px.scatter(
        sample,
        x="anomaly_score",
        y="transaction_amount",
        color="fraud_label",
        size="risk_score",
        hover_data=["transaction_id", "payment_channel", "authentication_type", "risk_tier"],
        color_discrete_map={"Fraud": "#dc2626", "Legitimate": "#2563eb"},
        title="Anomaly Score vs Transaction Amount",
        labels={"anomaly_score": "Anomaly score", "transaction_amount": "Transaction amount"},
    )
    fig_scatter.update_layout(height=410, margin=dict(l=20, r=20, t=70, b=20))
    right.plotly_chart(fig_scatter, use_container_width=True)

    left, right = st.columns(2)
    channel = fraud_rate_by(df, "payment_channel").sort_values("fraud_rate_pct", ascending=False)
    fig_channel = px.bar(
        channel,
        x="payment_channel",
        y="fraud_rate_pct",
        color="transactions",
        color_continuous_scale=["#93c5fd", "#1d4ed8"],
        title="Fraud Rate by Channel",
        labels={"payment_channel": "Payment channel", "fraud_rate_pct": "Fraud rate (%)"},
    )
    fig_channel.update_layout(height=380, margin=dict(l=20, r=20, t=70, b=20))
    left.plotly_chart(fig_channel, use_container_width=True)

    auth = fraud_rate_by(df, "authentication_type").sort_values("fraud_rate_pct", ascending=False)
    fig_auth = px.bar(
        auth,
        x="fraud_rate_pct",
        y="authentication_type",
        orientation="h",
        color="fraud_rate_pct",
        color_continuous_scale=["#99f6e4", "#0f766e"],
        title="Fraud Rate by Authentication Type",
        labels={"authentication_type": "Authentication type", "fraud_rate_pct": "Fraud rate (%)"},
    )
    fig_auth.update_layout(height=380, margin=dict(l=20, r=20, t=70, b=20))
    right.plotly_chart(fig_auth, use_container_width=True)

    numeric = df.select_dtypes(include="number").drop(columns=["transaction_id"], errors="ignore")
    corr = numeric.assign(fraud_flag=df["fraud_flag"].astype(int)).corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numeric Feature Correlation Matrix",
    )
    fig_corr.update_layout(height=660, margin=dict(l=20, r=20, t=70, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float, float]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    if len(thresholds):
        threshold_idx = min(best_idx, len(thresholds) - 1)
        threshold = float(thresholds[threshold_idx])
    else:
        threshold = 0.5
    return threshold, float(precision[best_idx]), float(recall[best_idx]), float(f1_scores[best_idx])


@st.cache_resource(show_spinner="Training classification and anomaly detection models...")
def train_fraud_models(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    model_df = df.copy()
    y = model_df["fraud_flag"].astype(int)
    X = model_df.drop(
        columns=[
            "fraud_flag",
            "fraud_label",
            "risk_score",
            "risk_tier",
            "expected_loss",
            "rules_alert",
        ],
        errors="ignore",
    )
    transaction_ids = X["transaction_id"].copy()
    X = X.drop(columns=["transaction_id"], errors="ignore")
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    def make_preprocess() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        transaction_ids,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    trained: dict[str, Pipeline] = {}
    metrics: list[ModelMetrics] = []
    test_predictions: dict[str, pd.DataFrame] = {}
    feature_importance: dict[str, pd.DataFrame] = {}

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocess", make_preprocess()), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        threshold, precision, recall, f1 = best_threshold(y_test.to_numpy(), probabilities)
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        metrics.append(
            ModelMetrics(
                model=name,
                roc_auc=float(roc_auc_score(y_test, probabilities)),
                pr_auc=float(average_precision_score(y_test, probabilities)),
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1,
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
            )
        )
        trained[name] = pipeline
        test_predictions[name] = pd.DataFrame(
            {
                "transaction_id": ids_test.to_numpy(),
                "actual_fraud": y_test.to_numpy(),
                "fraud_probability": probabilities,
                "prediction": predictions,
            }
        )

        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        fitted_model = pipeline.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            importance = fitted_model.feature_importances_
        else:
            importance = np.abs(fitted_model.coef_[0])
        feature_importance[name] = (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(12)
        )

    numeric_train = X_train[numeric_features]
    numeric_test = X_test[numeric_features]
    anomaly_model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=250,
                    contamination=float(y.mean()),
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    anomaly_model.fit(numeric_train)
    anomaly_score = -anomaly_model.decision_function(numeric_test)
    anomaly_metrics = {
        "roc_auc": float(roc_auc_score(y_test, anomaly_score)),
        "pr_auc": float(average_precision_score(y_test, anomaly_score)),
        "flag_rate": float((anomaly_score >= np.quantile(anomaly_score, 0.90)).mean()),
    }

    champion = max(metrics, key=lambda item: item.pr_auc)
    return {
        "metrics": metrics,
        "trained": trained,
        "test_predictions": test_predictions,
        "feature_importance": feature_importance,
        "champion": champion.model,
        "anomaly_metrics": anomaly_metrics,
        "feature_columns": X.columns.tolist(),
    }


def render_models(df: pd.DataFrame) -> dict[str, Any] | None:
    try:
        model_bundle = train_fraud_models(df)
    except Exception as exc:  # pragma: no cover - displayed in Streamlit runtime
        st.error(f"Model training could not run: {exc}")
        st.info("Install dependencies with: python -m pip install -r requirements.txt")
        return None

    metrics_df = pd.DataFrame([metric.__dict__ for metric in model_bundle["metrics"]])
    display = metrics_df.copy()
    for column in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
        display[column] = display[column].map(lambda x: f"{x:.4f}")
    display["threshold"] = display["threshold"].map(lambda x: f"{x:.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    champion_name = model_bundle["champion"]
    champion_metric = next(metric for metric in model_bundle["metrics"] if metric.model == champion_name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion Model", champion_name)
    c2.metric("Champion PR-AUC", f"{champion_metric.pr_auc:.4f}")
    c3.metric("Champion Recall", f"{champion_metric.recall:.4f}")
    c4.metric("Champion F1", f"{champion_metric.f1:.4f}")

    left, right = st.columns([1, 1.1])
    confusion = np.array([[champion_metric.tn, champion_metric.fp], [champion_metric.fn, champion_metric.tp]])
    fig_cm = px.imshow(
        confusion,
        text_auto=True,
        x=["Predicted Legitimate", "Predicted Fraud"],
        y=["Actual Legitimate", "Actual Fraud"],
        color_continuous_scale=["#eff6ff", "#dc2626"],
        title=f"{champion_name} Confusion Matrix",
    )
    fig_cm.update_layout(height=390, margin=dict(l=20, r=20, t=70, b=20))
    left.plotly_chart(fig_cm, use_container_width=True)

    importance = model_bundle["feature_importance"][champion_name].copy()
    importance["feature"] = importance["feature"].str.replace("numeric__", "", regex=False)
    importance["feature"] = importance["feature"].str.replace("categorical__", "", regex=False)
    fig_imp = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title=f"{champion_name} Top Drivers",
        color="importance",
        color_continuous_scale=["#99f6e4", "#0f766e"],
    )
    fig_imp.update_layout(height=390, margin=dict(l=20, r=20, t=70, b=20), yaxis_title="")
    right.plotly_chart(fig_imp, use_container_width=True)

    anomaly = model_bundle["anomaly_metrics"]
    st.markdown("#### Anomaly Detection")
    a1, a2, a3 = st.columns(3)
    a1.metric("Isolation Forest ROC-AUC", f"{anomaly['roc_auc']:.4f}")
    a2.metric("Isolation Forest PR-AUC", f"{anomaly['pr_auc']:.4f}")
    a3.metric("Top Decile Flag Rate", pct(anomaly["flag_rate"]))
    return model_bundle


def render_risk_scoring(df: pd.DataFrame) -> None:
    threshold = st.slider("Review threshold", min_value=0, max_value=100, value=75, step=1)
    review_queue = df[df["risk_score"] >= threshold].sort_values(
        ["risk_score", "transaction_amount"], ascending=[False, False]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions Above Threshold", f"{len(review_queue):,}")
    c2.metric("Review Queue Volume", money(float(review_queue["transaction_amount"].sum())) if len(review_queue) else "$0")
    c3.metric("Expected Exposure", money(float(review_queue["expected_loss"].sum())) if len(review_queue) else "$0")

    left, right = st.columns(2)
    fig_hist = px.histogram(
        df,
        x="risk_score",
        color="fraud_label",
        nbins=35,
        barmode="overlay",
        opacity=0.74,
        title="Risk Score Distribution",
        color_discrete_map={"Fraud": "#dc2626", "Legitimate": "#2563eb"},
        labels={"risk_score": "Risk score"},
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#111827")
    fig_hist.update_layout(height=390, margin=dict(l=20, r=20, t=70, b=20))
    left.plotly_chart(fig_hist, use_container_width=True)

    tier = (
        df.groupby("risk_tier", observed=False)
        .agg(transactions=("transaction_id", "count"), amount=("transaction_amount", "sum"), expected_loss=("expected_loss", "sum"))
        .reindex(["Low", "Watch", "High", "Critical"])
        .reset_index()
    )
    fig_tier = px.bar(
        tier,
        x="risk_tier",
        y="expected_loss",
        color="risk_tier",
        title="Expected Exposure by Risk Tier",
        color_discrete_map={
            "Low": "#16a34a",
            "Watch": "#ca8a04",
            "High": "#ea580c",
            "Critical": "#dc2626",
        },
        labels={"risk_tier": "Risk tier", "expected_loss": "Expected exposure"},
    )
    fig_tier.update_layout(height=390, showlegend=False, margin=dict(l=20, r=20, t=70, b=20))
    right.plotly_chart(fig_tier, use_container_width=True)

    columns = [
        "transaction_id",
        "transaction_amount",
        "payment_channel",
        "authentication_type",
        "anomaly_score",
        "device_risk_score",
        "transaction_velocity_score",
        "risk_score",
        "risk_tier",
        "expected_loss",
        "fraud_label",
    ]
    st.dataframe(review_queue[columns].head(250), use_container_width=True, hide_index=True)


def default_transaction(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "transaction_amount": float(df["transaction_amount"].median()),
        "login_attempts": int(df["login_attempts"].median()),
        "device_risk_score": float(df["device_risk_score"].median()),
        "transfer_frequency": int(df["transfer_frequency"].median()),
        "anomaly_score": float(df["anomaly_score"].quantile(0.75)),
        "account_age_days": int(df["account_age_days"].median()),
        "transaction_time_hour": int(df["transaction_time_hour"].median()),
        "failed_transactions_last_30d": int(df["failed_transactions_last_30d"].median()),
        "avg_monthly_balance": float(df["avg_monthly_balance"].median()),
        "daily_transaction_count": int(df["daily_transaction_count"].median()),
        "geo_distance_km": int(df["geo_distance_km"].median()),
        "session_duration_minutes": int(df["session_duration_minutes"].median()),
        "transaction_velocity_score": float(df["transaction_velocity_score"].median()),
        "payment_channel": str(df["payment_channel"].mode().iloc[0]),
        "authentication_type": str(df["authentication_type"].mode().iloc[0]),
        "card_present_flag": int(df["card_present_flag"].mode().iloc[0]),
        "international_transaction_flag": int(df["international_transaction_flag"].mode().iloc[0]),
        "suspicious_ip_flag": int(df["suspicious_ip_flag"].mode().iloc[0]),
    }


def render_predictive(df: pd.DataFrame, model_bundle: dict[str, Any] | None) -> None:
    defaults = default_transaction(df)
    st.markdown(
        """
        <div class="section-note">
            Score a new transaction with the champion classifier when dependencies are available.
            The rule-based risk score remains visible as an operational fallback.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("transaction_scoring_form"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Transaction amount", 0.0, 100000.0, defaults["transaction_amount"], step=100.0)
        anomaly = c2.slider("Anomaly score", 0.0, 1.0, defaults["anomaly_score"], 0.01)
        device = c3.slider("Device risk score", 0.0, 100.0, defaults["device_risk_score"], 0.1)

        c1, c2, c3 = st.columns(3)
        login_attempts = c1.number_input("Login attempts", 0, 25, defaults["login_attempts"])
        failed_30 = c2.number_input("Failed transactions last 30d", 0, 50, defaults["failed_transactions_last_30d"])
        velocity = c3.slider("Transaction velocity score", 0.0, 100.0, defaults["transaction_velocity_score"], 0.1)

        c1, c2, c3 = st.columns(3)
        channel = c1.selectbox("Payment channel", sorted(df["payment_channel"].unique()), index=sorted(df["payment_channel"].unique()).index(defaults["payment_channel"]))
        auth = c2.selectbox("Authentication type", sorted(df["authentication_type"].unique()), index=sorted(df["authentication_type"].unique()).index(defaults["authentication_type"]))
        hour = c3.slider("Transaction hour", 0, 23, defaults["transaction_time_hour"])

        c1, c2, c3 = st.columns(3)
        transfer_frequency = c1.number_input("Transfer frequency", 0, 120, defaults["transfer_frequency"])
        daily_count = c2.number_input("Daily transaction count", 0, 160, defaults["daily_transaction_count"])
        geo = c3.number_input("Geo distance km", 0, 20000, defaults["geo_distance_km"])

        c1, c2, c3 = st.columns(3)
        balance = c1.number_input("Average monthly balance", 0.0, 1000000.0, defaults["avg_monthly_balance"], step=1000.0)
        session = c2.number_input("Session duration minutes", 0, 240, defaults["session_duration_minutes"])
        account_age = c3.number_input("Account age days", 0, 10000, defaults["account_age_days"])

        c1, c2, c3 = st.columns(3)
        card_present = c1.checkbox("Card present", bool(defaults["card_present_flag"]))
        international = c2.checkbox("International transaction", bool(defaults["international_transaction_flag"]))
        suspicious_ip = c3.checkbox("Suspicious IP", bool(defaults["suspicious_ip_flag"]))
        st.form_submit_button("Score Transaction")

    scenario = pd.DataFrame(
        [
            {
                "transaction_amount": amount,
                "login_attempts": login_attempts,
                "device_risk_score": device,
                "transfer_frequency": transfer_frequency,
                "anomaly_score": anomaly,
                "account_age_days": account_age,
                "transaction_time_hour": hour,
                "failed_transactions_last_30d": failed_30,
                "avg_monthly_balance": balance,
                "daily_transaction_count": daily_count,
                "geo_distance_km": geo,
                "session_duration_minutes": session,
                "transaction_velocity_score": velocity,
                "payment_channel": channel,
                "authentication_type": auth,
                "card_present_flag": int(card_present),
                "international_transaction_flag": int(international),
                "suspicious_ip_flag": int(suspicious_ip),
            }
        ]
    )
    reference_columns = [
        column
        for column in df.columns
        if column
        not in {
            "risk_score",
            "risk_tier",
            "expected_loss",
            "rules_alert",
        }
    ]
    risk_reference = pd.concat(
        [
            df[reference_columns],
            scenario.assign(transaction_id=-1, fraud_flag=False, fraud_label="Scenario"),
        ],
        ignore_index=True,
    )
    scenario_with_risk = add_risk_features(risk_reference).tail(1)
    rule_score = float(scenario_with_risk["risk_score"].iloc[0])
    rule_tier = scenario_with_risk["risk_tier"].iloc[0]

    probability = None
    champion = None
    if model_bundle is not None:
        champion = model_bundle["champion"]
        model = model_bundle["trained"][champion]
        probability = float(model.predict_proba(scenario)[0, 1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Rule-Based Risk Score", f"{rule_score:.1f}/100")
    c2.metric("Risk Tier", rule_tier)
    if probability is None:
        c3.metric("Model Probability", "Install deps")
    else:
        c3.metric(f"{champion} Probability", pct(probability))


def render_financial_analytics(df: pd.DataFrame) -> None:
    left, right = st.columns([1.05, 1])

    tier_channel = (
        df.groupby(["payment_channel", "risk_tier"], observed=False)
        .agg(expected_loss=("expected_loss", "sum"), amount=("transaction_amount", "sum"))
        .reset_index()
    )
    fig_stack = px.bar(
        tier_channel,
        x="payment_channel",
        y="expected_loss",
        color="risk_tier",
        title="Expected Exposure Mix by Channel",
        color_discrete_map={
            "Low": "#16a34a",
            "Watch": "#ca8a04",
            "High": "#ea580c",
            "Critical": "#dc2626",
        },
        labels={"payment_channel": "Payment channel", "expected_loss": "Expected exposure", "risk_tier": "Risk tier"},
    )
    fig_stack.update_layout(height=430, margin=dict(l=20, r=20, t=70, b=20))
    left.plotly_chart(fig_stack, use_container_width=True)

    amount_band = df.copy()
    quantile_count = max(1, min(5, int(amount_band["transaction_amount"].nunique())))
    labels = ["Very Low", "Low", "Mid", "High", "Very High"]
    amount_band["amount_band"] = pd.qcut(
        amount_band["transaction_amount"],
        q=quantile_count,
        labels=False,
        duplicates="drop",
    ).map({index: label for index, label in enumerate(labels[:quantile_count])})
    band = (
        amount_band.groupby("amount_band", observed=False)
        .agg(transactions=("transaction_id", "count"), fraud_rate=("fraud_flag", "mean"), amount=("transaction_amount", "sum"))
        .reset_index()
    )
    band["fraud_rate_pct"] = band["fraud_rate"] * 100
    fig_band = px.line(
        band,
        x="amount_band",
        y="fraud_rate_pct",
        markers=True,
        title="Fraud Rate Across Transaction Amount Bands",
        labels={"amount_band": "Amount band", "fraud_rate_pct": "Fraud rate (%)"},
    )
    fig_band.update_traces(line_color="#dc2626", marker_size=10)
    fig_band.update_layout(height=430, margin=dict(l=20, r=20, t=70, b=20))
    right.plotly_chart(fig_band, use_container_width=True)

    portfolio = (
        df.groupby(["payment_channel", "authentication_type"], observed=False)
        .agg(
            transactions=("transaction_id", "count"),
            amount=("transaction_amount", "sum"),
            fraud_rate=("fraud_flag", "mean"),
            expected_loss=("expected_loss", "sum"),
        )
        .reset_index()
        .sort_values("expected_loss", ascending=False)
    )
    portfolio["fraud_rate"] = portfolio["fraud_rate"].map(lambda x: f"{x * 100:.2f}%")
    portfolio["amount"] = portfolio["amount"].map(money)
    portfolio["expected_loss"] = portfolio["expected_loss"].map(money)
    st.dataframe(portfolio, use_container_width=True, hide_index=True)


def main() -> None:
    df = load_data()
    render_header()
    filtered = apply_sidebar_filters(df)
    if filtered.empty:
        st.warning("No transactions match the selected filters.")
        return

    render_kpis(filtered)

    overview, eda, models, scoring, predictive, financial = st.tabs(
        [
            "Executive Overview",
            "Exploratory Data Analysis",
            "Fraud Detection Models",
            "Risk Scoring Systems",
            "Predictive Analytics",
            "Financial Analytics",
        ]
    )
    with overview:
        render_overview(filtered)
    with eda:
        render_eda(filtered)
    with models:
        model_bundle = render_models(df)
    with scoring:
        render_risk_scoring(filtered)
    with predictive:
        model_bundle = render_models(df) if "model_bundle" not in locals() else model_bundle
        render_predictive(df, model_bundle)
    with financial:
        render_financial_analytics(filtered)

    st.markdown(f'<div class="footer">Created by {CREATOR}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
