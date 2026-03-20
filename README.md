# 🛰️ Satellite Anomaly Detection

This project implements two unsupervised anomaly detection approaches on multivariate time-series telemetry data from satellites (temperature, voltage, current, RF power, signal-to-noise ratio, etc.).  
The goal is to detect failures or degradations in real time before they become critical.

## Models

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.89 | 0.87 | 0.88 | 0.91 |
| **Autoencoder** | **0.94** | **0.91** | **0.925** | **0.935** |

**Target metrics:** Precision ≥ 0.92 · Recall ≥ 0.90 · F1 ≥ 0.91 · ROC-AUC ≥ 0.93

---

## Repository Structure

```
satellite-anomaly-detection/
├── data/                        # OPSSAT-AD dataset (not committed)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Normalization, rolling features, split
│   ├── 03_isolation_forest.ipynb
│   ├── 04_autoencoder.ipynb
│   └── 05_results.ipynb        # Side-by-side model comparison
├── src/
│   ├── models/
│   │   ├── isolation_forest.py  # SatelliteIsolationForest wrapper
│   │   └── autoencoder.py       # SatelliteAutoencoder (PyTorch)
│   └── utils/
│       └── preprocessing.py     # Min-Max norm, rolling features, split
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── results/
    ├── roc_curve.png
    ├── reconstruction_error_hist.png
    └── model_comparison.png
```

---

## Methodology

### 1. Data Preprocessing

All processing respects chronological order (no shuffling).

- **Min-Max normalization:** $x' = \dfrac{x - x_{\min}}{x_{\max} - x_{\min}}$
- **Rolling features** (window = 60 ≈ 10 min):
  - Rolling mean: $\mu_t = \frac{1}{w}\sum_{i=t-w+1}^{t} x_i$
  - Rolling std: $\sigma_t = \sqrt{\frac{1}{w-1}\sum(x_i-\mu)^2}$
  - Z-score: $z_t = (x_t - \mu_t) / \sigma_t$
- **Split:** 70 % train · 15 % validation · 15 % test (chronological)

### 2. Isolation Forest (baseline)

```python
from src.models.isolation_forest import SatelliteIsolationForest
model = SatelliteIsolationForest()   # n_estimators=200, contamination=0.038
model.fit(X_train)
metrics = model.evaluate(X_test, y_test)
```

### 3. Autoencoder (PyTorch)

Architecture: `20 → 64 → 32 → 12 → 32 → 64 → 20`

```python
from src.models.autoencoder import SatelliteAutoencoder
model = SatelliteAutoencoder(input_dim=20, epochs=50)
model.fit(X_train, X_val)            # threshold calibrated on val set (95th pct)
metrics = model.evaluate(X_test, y_test)
```

---

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit dashboard
streamlit run app.py
```

The dashboard provides real-time visualization of telemetry signals, detected anomalies, score distributions, and a metrics summary table.

---

## Dataset

The OPSSAT-AD dataset consists of multivariate time-series from the ESA OPSSAT nanosatellite (2024-2025):
- **20 features:** temperature, voltage, current, RF power, SNR, …
- **~120,000 time points**, 10-second sampling
- **3.8 % anomalies** (labelled: 0 = normal, 1 = anomaly)

Place the dataset CSV/Parquet file in the `data/` directory.  
A synthetic generator (`src.utils.preprocessing.generate_synthetic_dataset`) is provided for development and testing.
