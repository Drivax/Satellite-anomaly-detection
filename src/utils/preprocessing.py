"""
Preprocessing utilities for the satellite anomaly detection pipeline.

Includes:
- Min-Max normalization
- Rolling statistical features (mean, std, z-score) with window=60
- Chronological train/validation/test split (70/15/15)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ROLLING_WINDOW = 60  # ~10 minutes at 1 sample/10 s
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15  (implicit)

N_FEATURES = 20

FEATURE_COLUMNS = [
    "temperature_1", "temperature_2", "temperature_3",
    "voltage_1", "voltage_2", "voltage_3",
    "current_1", "current_2", "current_3",
    "rf_power_1", "rf_power_2",
    "snr_1", "snr_2",
    "pressure_1", "pressure_2",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y",
]


def minmax_normalize(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Apply Min-Max normalization: x' = (x - x_min) / (x_max - x_min).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must be sorted chronologically).
    feature_cols : list of str
        Column names to normalize.

    Returns
    -------
    df_norm : pd.DataFrame
        Dataframe with normalized features.
    scaler : MinMaxScaler
        Fitted scaler (use to transform future data).
    """
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_norm, scaler


def add_rolling_features(df: pd.DataFrame, feature_cols: list,
                         window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Compute rolling mean, rolling std, and z-score for each feature column.

    Rolling mean:  mu_t  = mean(x_{t-w+1}, ..., x_t)
    Rolling std:   sig_t = std(x_{t-w+1}, ..., x_t)  (ddof=1)
    Z-score:       z_t   = (x_t - mu_t) / sig_t

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (sorted chronologically, already normalized).
    feature_cols : list of str
        Base feature columns.
    window : int
        Rolling window size (default 60).

    Returns
    -------
    pd.DataFrame
        Original dataframe augmented with rolling features.
        NaN rows at the beginning (< window) are filled with 0.
    """
    df_out = df.copy()
    for col in feature_cols:
        roll = df_out[col].rolling(window=window, min_periods=window)
        mu = roll.mean()
        sigma = roll.std(ddof=1)
        df_out[f"{col}_roll_mean"] = mu
        df_out[f"{col}_roll_std"] = sigma
        df_out[f"{col}_zscore"] = (df_out[col] - mu) / sigma.replace(0, 1e-8)

    # Fill NaN values introduced by the rolling window with 0
    roll_cols = [c for c in df_out.columns if c not in df.columns]
    df_out[roll_cols] = df_out[roll_cols].fillna(0)
    return df_out


def chronological_split(df: pd.DataFrame,
                         train_ratio: float = TRAIN_RATIO,
                         val_ratio: float = VAL_RATIO) -> tuple:
    """
    Split a time-ordered dataframe into train, validation and test sets.

    No shuffling is performed to respect temporal order.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset sorted by time.
    train_ratio : float
        Fraction of data for training (default 0.70).
    val_ratio : float
        Fraction of data for validation (default 0.15).

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def preprocess_pipeline(df: pd.DataFrame,
                         feature_cols: list = None,
                         window: int = ROLLING_WINDOW) -> tuple:
    """
    Full preprocessing pipeline:
      1. Min-Max normalization
      2. Rolling feature computation
      3. Chronological 70/15/15 split

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with a 'label' column (0=normal, 1=anomaly).
    feature_cols : list of str, optional
        Feature columns to use (default: FEATURE_COLUMNS).
    window : int
        Rolling window size.

    Returns
    -------
    train, val, test : pd.DataFrame
        Preprocessed splits.
    scaler : MinMaxScaler
        Fitted Min-Max scaler.
    all_feature_cols : list of str
        All feature columns after adding rolling features.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    df_norm, scaler = minmax_normalize(df, feature_cols)
    df_feat = add_rolling_features(df_norm, feature_cols, window=window)
    train, val, test = chronological_split(df_feat)

    all_feature_cols = feature_cols + [
        c for c in df_feat.columns
        if c not in df.columns
    ]
    return train, val, test, scaler, all_feature_cols


def generate_synthetic_dataset(n_samples: int = 120_000,
                                anomaly_ratio: float = 0.038,
                                n_features: int = N_FEATURES,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic multivariate time-series dataset that mimics the
    OPSSAT-AD dataset structure for testing purposes.

    Parameters
    ----------
    n_samples : int
        Total number of time points.
    anomaly_ratio : float
        Fraction of anomalous samples (~3.8 %).
    n_features : int
        Number of telemetry features.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with timestamps, feature columns and 'label' column.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="10s")

    # Normal data: correlated multivariate Gaussian
    data = rng.standard_normal((n_samples, n_features))
    # Add mild temporal trend to some features
    trend = np.linspace(0, 0.5, n_samples)
    data[:, :3] += trend[:, None]

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_idx] = 1
    # Anomalies: large deviation from normal
    data[anomaly_idx] += rng.uniform(3, 6, size=(n_anomalies, n_features))

    cols = FEATURE_COLUMNS[:n_features]
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "timestamp", timestamps)
    df["label"] = labels
    return df
