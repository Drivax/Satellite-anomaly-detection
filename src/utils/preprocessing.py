"""
Preprocessing utilities for the satellite anomaly detection pipeline.

Includes:
- Min-Max normalization
- Orbital phase encoding (LEO ~90 min cyclical features)
- Eclipse detection from SNR
- Physics-based features (thermal gradients, power estimates, gyro/mag magnitude)
- Domain-weighted feature importance for sensor criticality
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

# --- Satellite domain constants ----------------------------------------
ORBITAL_PERIOD_SAMPLES = 540   # LEO ~90 min at 10 s sampling
ECLIPSE_SNR_PERCENTILE = 25    # SNR below this percentile → eclipse

SENSOR_CRITICALITY = {
    "rf_power": 1.5,
    "snr": 1.5,
    "voltage": 1.3,
    "current": 1.3,
    "power_estimate": 1.3,
    "temperature": 1.0,
    "thermal_gradient": 1.0,
    "pressure": 1.0,
    "orbital": 1.0,
    "eclipse": 1.2,
    "gyro": 0.8,
    "mag": 0.8,
}


def minmax_normalize(df: pd.DataFrame,
                     feature_cols: list,
                     scaler: MinMaxScaler = None) -> tuple:
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
    scaler = scaler or MinMaxScaler()
    df_norm = df.copy()
    if hasattr(scaler, "n_features_in_"):
        df_norm[feature_cols] = scaler.transform(df[feature_cols])
    else:
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


def add_orbital_features(df: pd.DataFrame,
                         orbital_period: int = ORBITAL_PERIOD_SAMPLES,
                         eclipse_percentile: int = ECLIPSE_SNR_PERCENTILE,
                         start_index: int = 0,
                         eclipse_threshold: float = None) -> pd.DataFrame:
    """
    Add cyclical orbital phase encoding and eclipse detection.

    LEO satellites orbit Earth in ~90 minutes (540 samples at 10 s).
    The orbital phase is encoded as sin/cos to avoid discontinuity at
    period boundaries.  Eclipse is detected when mean SNR drops below
    the given percentile (RF link degrades during Earth shadow).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (sorted chronologically, already normalized).
    orbital_period : int
        Orbital period in samples (default 540).
    eclipse_percentile : int
        Percentile threshold for eclipse detection (default 25).

    Returns
    -------
    pd.DataFrame
        Dataframe with added orbital_cos, orbital_sin, and eclipse columns.
    """
    df_out = df.copy()
    n = len(df_out)

    # Cyclical orbital phase encoding
    time_in_orbit = (np.arange(start_index, start_index + n)
                     % orbital_period)
    df_out["orbital_cos"] = np.cos(2 * np.pi * time_in_orbit / orbital_period)
    df_out["orbital_sin"] = np.sin(2 * np.pi * time_in_orbit / orbital_period)

    # Eclipse detection based on SNR
    snr_cols = [c for c in df_out.columns if c.startswith("snr_")]
    if snr_cols:
        mean_snr = df_out[snr_cols].mean(axis=1)
        threshold = eclipse_threshold
        if threshold is None:
            threshold = mean_snr.quantile(eclipse_percentile / 100)
        df_out["eclipse"] = (mean_snr < threshold).astype(float)

    return df_out


def compute_eclipse_threshold(df: pd.DataFrame,
                              eclipse_percentile: int = ECLIPSE_SNR_PERCENTILE
                              ) -> float:
    """Compute the eclipse threshold from training data only."""
    snr_cols = [c for c in df.columns if c.startswith("snr_")]
    if not snr_cols:
        return None
    mean_snr = df[snr_cols].mean(axis=1)
    return float(mean_snr.quantile(eclipse_percentile / 100))


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-based derived features specific to satellite telemetry.

    - Thermal gradients between adjacent temperature sensors
      (heat transfer between subsystems)
    - Power estimates  P = V × I  for each voltage/current rail
    - Gyroscope angular rate magnitude  ||ω|| = √(ωx² + ωy² + ωz²)
    - Magnetic field magnitude  ||B|| = √(Bx² + By²)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (sorted chronologically, already normalized).

    Returns
    -------
    pd.DataFrame
        Dataframe with added physics-based columns.
    """
    df_out = df.copy()

    # Thermal gradients
    if "temperature_1" in df_out.columns and "temperature_2" in df_out.columns:
        df_out["thermal_gradient_12"] = (
            df_out["temperature_1"] - df_out["temperature_2"]
        )
    if "temperature_2" in df_out.columns and "temperature_3" in df_out.columns:
        df_out["thermal_gradient_23"] = (
            df_out["temperature_2"] - df_out["temperature_3"]
        )

    # Power estimates  P = V × I
    for i in range(1, 4):
        v_col, c_col = f"voltage_{i}", f"current_{i}"
        if v_col in df_out.columns and c_col in df_out.columns:
            df_out[f"power_estimate_{i}"] = df_out[v_col] * df_out[c_col]

    # Gyroscope angular rate magnitude
    gyro_cols = [c for c in df_out.columns if c.startswith("gyro_")]
    if gyro_cols:
        df_out["gyro_magnitude"] = np.sqrt(
            (df_out[gyro_cols] ** 2).sum(axis=1)
        )

    # Magnetic field magnitude
    mag_cols = [c for c in df_out.columns if c.startswith("mag_")]
    if mag_cols:
        df_out["mag_magnitude"] = np.sqrt(
            (df_out[mag_cols] ** 2).sum(axis=1)
        )

    return df_out


def build_feature_weights(feature_cols: list) -> np.ndarray:
    """
    Build a weight vector for domain-weighted loss based on sensor criticality.

    Weights reflect operational importance of each subsystem:
      RF power, SNR     : 1.5  (mission-critical communications)
      Voltage, Current  : 1.3  (power subsystem)
      Eclipse indicator : 1.2  (orbital context)
      Temperature, etc. : 1.0  (standard)
      Gyro, Mag         : 0.8  (less critical for most missions)

    Parameters
    ----------
    feature_cols : list of str
        Ordered list of feature column names.

    Returns
    -------
    np.ndarray of shape (len(feature_cols),)
        Per-feature weight vector.
    """
    weights = np.ones(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        for prefix, w in SENSOR_CRITICALITY.items():
            if col.startswith(prefix):
                weights[i] = w
                break
    return weights


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
    1. Chronological split
    2. Min-Max normalization fitted on train only
    3. Orbital phase encoding & eclipse detection
    4. Physics-based feature derivation
    5. Rolling feature computation

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

    train_raw, val_raw, test_raw = chronological_split(df)

    # Fit scaler on NORMAL training data only so anomalies fall outside [0,1]
    train_normal_mask = train_raw['label'] == 0
    scaler = MinMaxScaler().fit(train_raw.loc[train_normal_mask, feature_cols])
    df_norm, _ = minmax_normalize(df, feature_cols, scaler=scaler)

    eclipse_threshold = compute_eclipse_threshold(
        df_norm.iloc[:len(train_raw)]
    )
    df_domain = add_orbital_features(
        df_norm,
        eclipse_threshold=eclipse_threshold,
    )
    df_domain = add_physics_features(df_domain)
    df_feat = add_rolling_features(df_domain, feature_cols, window=window)
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

    # Inject anomalies as short temporal episodes rather than isolated points.
    n_anomalies = int(n_samples * anomaly_ratio)
    labels = np.zeros(n_samples, dtype=int)

    feature_index = {
        col: idx for idx, col in enumerate(FEATURE_COLUMNS[:n_features])
    }
    anomaly_types = ["thermal", "power", "rf", "attitude", "mixed"]
    anomaly_probs = [0.25, 0.25, 0.20, 0.20, 0.10]

    filled = 0
    occupied = np.zeros(n_samples, dtype=bool)
    while filled < n_anomalies:
        remaining = n_anomalies - filled
        seg_len = int(min(rng.integers(12, 36), remaining))
        start = int(rng.integers(0, max(1, n_samples - seg_len)))
        stop = min(n_samples, start + seg_len)
        if occupied[start:stop].any():
            continue

        occupied[start:stop] = True
        labels[start:stop] = 1
        filled += stop - start

        idx = np.arange(start, stop)
        phase = np.linspace(0, 1, len(idx))
        triangle = 1.0 - np.abs(2.0 * phase - 1.0)
        smooth = np.sin(np.pi * phase)
        anomaly_type = rng.choice(anomaly_types, p=anomaly_probs)

        if anomaly_type == "thermal":
            for col in ("temperature_1", "temperature_2", "temperature_3"):
                if col in feature_index:
                    data[idx, feature_index[col]] += rng.uniform(1.5, 3.0) * smooth
        elif anomaly_type == "power":
            for col in ("voltage_1", "voltage_2", "voltage_3"):
                if col in feature_index:
                    data[idx, feature_index[col]] -= rng.uniform(1.5, 2.5) * smooth
            for col in ("current_1", "current_2", "current_3"):
                if col in feature_index:
                    data[idx, feature_index[col]] += rng.uniform(1.0, 2.0) * triangle
        elif anomaly_type == "rf":
            for col in ("rf_power_1", "rf_power_2", "snr_1", "snr_2"):
                if col in feature_index:
                    data[idx, feature_index[col]] -= rng.uniform(2.0, 4.0) * smooth
        elif anomaly_type == "attitude":
            oscillation = np.sin(np.linspace(0, 3 * np.pi, len(idx)))
            for col in ("gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y"):
                if col in feature_index:
                    data[idx, feature_index[col]] += rng.uniform(1.5, 3.0) * oscillation
        else:
            active_cols = rng.choice(
                list(feature_index.values()),
                size=max(3, n_features // 4),
                replace=False,
            )
            shifts = rng.uniform(1.0, 2.5, size=len(active_cols))
            signs = rng.choice([-1.0, 1.0], size=len(active_cols))
            data[np.ix_(idx, active_cols)] += (
                (signs * shifts)[None, :] * smooth[:, None]
                + rng.normal(0, 0.3, size=(len(idx), len(active_cols)))
            )

        data[idx] += rng.normal(0, 0.05, size=(len(idx), n_features))

    cols = FEATURE_COLUMNS[:n_features]
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "timestamp", timestamps)
    df["label"] = labels
    return df
