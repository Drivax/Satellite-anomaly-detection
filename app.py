"""
Streamlit dashboard for satellite anomaly detection.

Features:
- Real-time signal visualization
- Anomaly overlay on time series
- Isolation Forest and Autoencoder score plots
- Summary metrics table

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch

from src.utils.preprocessing import (
    generate_synthetic_dataset,
    preprocess_pipeline,
    FEATURE_COLUMNS,
)
from src.models.isolation_forest import SatelliteIsolationForest
from src.models.autoencoder import SatelliteAutoencoder

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Satellite Anomaly Detection Dashboard")
st.markdown(
    "Real-time visualization of satellite telemetry signals and detected "
    "anomalies using **Isolation Forest** and **Autoencoder** models."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

n_samples = st.sidebar.slider(
    "Number of time points", min_value=1_000, max_value=20_000,
    value=5_000, step=500,
)
anomaly_ratio = st.sidebar.slider(
    "Anomaly ratio", min_value=0.01, max_value=0.10,
    value=0.038, step=0.005, format="%.3f",
)
seed = st.sidebar.number_input("Random seed", value=42, min_value=0)

selected_feature = st.sidebar.selectbox(
    "Feature to visualize", FEATURE_COLUMNS[:10], index=0,
)

run_models = st.sidebar.button("▶ Run Detection", type="primary")

# ──────────────────────────────────────────────────────────────────────────────
# Data generation & preprocessing
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Generating synthetic dataset…")
def load_data(n: int, ratio: float, s: int):
    df = generate_synthetic_dataset(
        n_samples=n, anomaly_ratio=ratio, seed=s
    )
    return df


@st.cache_data(show_spinner="Preprocessing…")
def preprocess(df: pd.DataFrame):
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    train, val, test, scaler, all_cols = preprocess_pipeline(
        df, feature_cols=feature_cols
    )
    return train, val, test, scaler, all_cols


df_raw = load_data(n_samples, anomaly_ratio, int(seed))
train, val, test, scaler, all_feature_cols = preprocess(df_raw)

base_feature_cols = [c for c in FEATURE_COLUMNS if c in df_raw.columns]
X_train = train[base_feature_cols].values
X_val = val[base_feature_cols].values
X_test = test[base_feature_cols].values
y_test = test["label"].values

# ──────────────────────────────────────────────────────────────────────────────
# Raw signal plot
# ──────────────────────────────────────────────────────────────────────────────
st.subheader(f"📡 Raw Signal: {selected_feature}")

plot_df = df_raw[["timestamp", selected_feature, "label"]].copy()
anomaly_mask = plot_df["label"] == 1

fig_signal = go.Figure()
fig_signal.add_trace(
    go.Scatter(
        x=plot_df["timestamp"],
        y=plot_df[selected_feature],
        mode="lines",
        name="Normal",
        line=dict(color="#1f77b4", width=1),
    )
)
if anomaly_mask.any():
    fig_signal.add_trace(
        go.Scatter(
            x=plot_df.loc[anomaly_mask, "timestamp"],
            y=plot_df.loc[anomaly_mask, selected_feature],
            mode="markers",
            name="Anomaly (true)",
            marker=dict(color="red", size=5, symbol="x"),
        )
    )
fig_signal.update_layout(
    xaxis_title="Time",
    yaxis_title=selected_feature,
    height=350,
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_signal, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model training and evaluation
# ──────────────────────────────────────────────────────────────────────────────
if run_models:
    col1, col2 = st.columns(2)

    # ── Isolation Forest ──────────────────────────────────────────────────────
    with col1:
        st.subheader("🌲 Isolation Forest")
        with st.spinner("Training Isolation Forest…"):
            if_model = SatelliteIsolationForest()
            if_model.fit(X_train)
            if_metrics = if_model.evaluate(X_test, y_test)
            if_scores = if_model.anomaly_scores(X_test)
            if_preds = if_model.predict(X_test)

        m = if_metrics
        st.metric("Precision", f"{m['precision']:.3f}")
        st.metric("Recall", f"{m['recall']:.3f}")
        st.metric("F1-score", f"{m['f1']:.3f}")
        st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

        # Score distribution
        fig_if = px.histogram(
            x=if_scores,
            color=y_test.astype(str),
            barmode="overlay",
            nbins=80,
            labels={"x": "Anomaly Score", "color": "True Label"},
            title="IF Score Distribution",
            color_discrete_map={"0": "#1f77b4", "1": "#d62728"},
        )
        fig_if.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_if, use_container_width=True)

    # ── Autoencoder ───────────────────────────────────────────────────────────
    with col2:
        st.subheader("🤖 Autoencoder")
        with st.spinner("Training Autoencoder (50 epochs)…"):
            ae_model = SatelliteAutoencoder(
                input_dim=len(base_feature_cols), epochs=50
            )
            ae_model.fit(X_train, X_val)
            ae_metrics = ae_model.evaluate(X_test, y_test)
            ae_scores = ae_model.anomaly_scores(X_test)
            ae_preds = ae_model.predict(X_test)

        m = ae_metrics
        st.metric("Precision", f"{m['precision']:.3f}")
        st.metric("Recall", f"{m['recall']:.3f}")
        st.metric("F1-score", f"{m['f1']:.3f}")
        st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")

        # Reconstruction error distribution
        fig_ae = px.histogram(
            x=ae_scores,
            color=y_test.astype(str),
            barmode="overlay",
            nbins=80,
            labels={"x": "Reconstruction Error", "color": "True Label"},
            title="AE Reconstruction Error Distribution",
            color_discrete_map={"0": "#1f77b4", "1": "#d62728"},
        )
        fig_ae.add_vline(
            x=ae_model.threshold_, line_dash="dash",
            line_color="orange", annotation_text="threshold",
        )
        fig_ae.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ae, use_container_width=True)

    # ── Test set anomaly overlay ───────────────────────────────────────────────
    st.subheader("🔍 Detected Anomalies on Test Set")
    test_timestamps = test["timestamp"].values
    test_signal = test[selected_feature].values

    fig_det = go.Figure()
    fig_det.add_trace(
        go.Scatter(
            x=test_timestamps, y=test_signal,
            mode="lines", name="Signal",
            line=dict(color="#1f77b4", width=1),
        )
    )
    # True anomalies
    true_mask = y_test == 1
    if true_mask.any():
        fig_det.add_trace(
            go.Scatter(
                x=test_timestamps[true_mask], y=test_signal[true_mask],
                mode="markers", name="True Anomaly",
                marker=dict(color="red", size=6, symbol="x"),
            )
        )
    # IF predictions
    if_mask = if_preds == 1
    if if_mask.any():
        fig_det.add_trace(
            go.Scatter(
                x=test_timestamps[if_mask], y=test_signal[if_mask],
                mode="markers", name="IF Detected",
                marker=dict(color="orange", size=5, symbol="circle-open"),
            )
        )
    # AE predictions
    ae_mask = ae_preds == 1
    if ae_mask.any():
        fig_det.add_trace(
            go.Scatter(
                x=test_timestamps[ae_mask], y=test_signal[ae_mask],
                mode="markers", name="AE Detected",
                marker=dict(color="green", size=5, symbol="diamond-open"),
            )
        )
    fig_det.update_layout(
        xaxis_title="Time", yaxis_title=selected_feature,
        height=400,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig_det, use_container_width=True)

    # ── Metrics comparison table ───────────────────────────────────────────────
    st.subheader("📊 Metrics Summary")
    metrics_df = pd.DataFrame(
        {
            "Model": ["Isolation Forest", "Autoencoder"],
            "Precision": [if_metrics["precision"], ae_metrics["precision"]],
            "Recall": [if_metrics["recall"], ae_metrics["recall"]],
            "F1-score": [if_metrics["f1"], ae_metrics["f1"]],
            "ROC-AUC": [if_metrics["roc_auc"], ae_metrics["roc_auc"]],
        }
    ).set_index("Model")
    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    # Target metrics reference
    st.info(
        "**Target metrics** — "
        "Precision ≥ 0.92 | Recall ≥ 0.90 | F1 ≥ 0.91 | ROC-AUC ≥ 0.93"
    )

else:
    st.info("👈 Click **Run Detection** in the sidebar to train models and see results.")
