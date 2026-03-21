# Satellite Anomaly Detection

## Project Objective

This project implements two unsupervised anomaly detection approaches on multivariate time-series telemetry data from satellites: temperature, voltage, current, RF power, signal-to-noise ratio, and related hardware indicators. The objective is to detect failures or performance degradations in real time, before they become operationally critical.

Satellite telemetry is inherently high-dimensional, noisy, and temporally correlated. Classical threshold-based monitoring fails to capture cross-channel dependencies or gradual drifts. The approach here treats anomaly detection as a density estimation and reconstruction problem, using both a statistical baseline and a deep learning model trained exclusively on normal operating data.

Two models are implemented and compared:

- **Isolation Forest** — statistical baseline, tree-based unsupervised method
- **Autoencoder** — deep learning model with **satellite domain-specific** feature engineering and weighted loss; anomalies surface through elevated reconstruction error

Target performance metrics:

| Metric | Target |
|---|---|
| Precision | >= 0.92 |
| Recall | >= 0.90 |
| F1-score | >= 0.91 |
| ROC-AUC | >= 0.93 |

These results are measured on the OPSSAT-AD dataset, real ESA telemetry data from 2024-2025.

---

## Dataset

The OPSSAT-AD dataset contains multivariate time-series recorded aboard the ESA OPSSAT nanosatellite. Each observation is a 20-dimensional vector sampled at 10-second intervals, covering:

- Thermal sensors (on-board computer, battery, solar panels)
- Power bus voltage and current rails
- RF transmit power and received signal-to-noise ratio
- Attitude control and reaction wheel housekeeping

Key statistics:

- Total time points: approximately 120,000
- Number of features: 20
- Anomaly rate: 3.8 % (label 0 = normal, label 1 = anomaly)
- Anomalies include: thermal excursions, power rail drops, RF link degradation, attitude instability

Place the dataset CSV or Parquet file in the `data/` directory. The file is not committed to the repository.

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
│       └── preprocessing.py     # Normalization, rolling features, split
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

All processing strictly respects the chronological order of the time series. No shuffling is applied at any stage, as shuffling would introduce data leakage from the future into the training set and invalidate temporal dependencies.

**Min-Max normalization** is applied per feature to bring all channels into [0, 1]:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Normalization parameters are computed on the training set only and applied identically to the validation and test sets, preventing any leakage from held-out data.

**Rolling features** are computed with a window of $w = 60$ points (equivalent to 10 minutes at 10-second sampling). For each feature and each time step $t$:

Rolling mean:

$$ \mu_t = \frac{1}{w} \sum_{i=t-w+1}^{t} x_i $$

Rolling standard deviation (unbiased, Bessel-corrected):

$$ \sigma_t = \sqrt{\frac{1}{w-1} \sum_{i=t-w+1}^{t} (x_i - \mu_t)^2} $$

Z-score (local standardization relative to recent history):

$$z_t = \frac{x_t - \mu_t}{\sigma_t}$$

These rolling statistics enrich each observation with local temporal context, allowing both models to detect deviations relative to the recent operating regime rather than global statistics — which is essential for detecting gradual drifts.

**Orbital phase encoding (LEO cyclical features):**

Low Earth Orbit satellites complete one orbit approximately every 90 minutes. At 10-second sampling, this corresponds to a period of $T = 540$ samples. To inject orbital context into the feature space without introducing a discontinuity at period boundaries, the orbital phase is encoded with sine and cosine:

$$\text{orbital_cos}_t = \cos\!\left(\frac{2\pi\,(t \bmod T)}{T}\right)$$

$$\text{orbital_sin}_t = \sin\!\left(\frac{2\pi\,(t \bmod T)}{T}\right)$$

This pair uniquely identifies the position within each orbit and enables the model to learn phase-dependent normal behavior (e.g., thermal cycles caused by sun/shadow transitions).

**Eclipse detection:**

When a LEO satellite enters Earth's shadow, its RF link quality degrades and solar power drops. Eclipse periods are detected by thresholding the mean SNR at the 25th percentile:

$$\text{eclipse}_t = \mathbb{1}\!\left[\overline{\text{SNR}}_t < Q_{25}(\overline{\text{SNR}})\right]$$

This binary feature allows the model to distinguish anomalies from expected eclipse-induced signal drops.

**Physics-based derived features:**

| Feature | Formula | Rationale |
|---|---|---|
| Thermal gradient | $\Delta T_{ij} = T_i - T_j$ | Heat transfer between adjacent subsystems |
| Power estimate | $P_k = V_k \times I_k$ | Ohmic power on each rail; deviations indicate faults |
| Gyro magnitude | $\|\omega\| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$ | Total angular rate; spikes indicate attitude instability |
| Mag magnitude | $\|B\| = \sqrt{B_x^2 + B_y^2}$ | Geomagnetic field intensity; anomalies from sensor faults |

**Chronological split:**

| Set | Proportion | Purpose |
|---|---|---|
| Train | 70 % | Model fitting |
| Validation | 15 % | Threshold calibration, early stopping |
| Test | 15 % | Final evaluation |

---

### 2. Baseline Model: Isolation Forest

Isolation Forest (Liu et al., 2008) is an unsupervised anomaly detection algorithm operating on the principle that anomalies are statistically rare and lie in sparse regions of the feature space. They are therefore easier to isolate through random recursive partitioning.

**Construction:** An ensemble of $T$ isolation trees is built. Each tree is grown by randomly selecting a feature and a random split value within the observed range of that feature, recursively, until each point is isolated in its own leaf.

**Anomaly score:** The isolation depth $h(x)$ for a point $x$ is the number of splits required to isolate it. Normal points require many splits (they are deep in dense clusters); anomalies require few splits (they are separated early in sparse regions). The normalized anomaly score is:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

where:
- $E[h(x)]$ is the expected isolation depth averaged over all $T$ trees
- $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$ is the expected depth of an unsuccessful search in a binary search tree over $n$ samples, with $H(k) = \ln(k) + \gamma$ (Euler-Mascheroni constant $\gamma \approx 0.5772$)
- $s(x) \to 1$ indicates an anomaly; $s(x) \to 0$ indicates a normal point; $s(x) \approx 0.5$ is ambiguous

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Sufficient for score stability |
| `contamination` | 0.038 | Matches the empirical anomaly rate |
| `max_samples` | 0.8 | Sub-sampling reduces variance, improves isolation |

```python
from src.models.isolation_forest import SatelliteIsolationForest
model = SatelliteIsolationForest()
model.fit(X_train)
metrics = model.evaluate(X_test, y_test)
```

---

### 3. Advanced Model: Autoencoder (PyTorch)

An autoencoder is a neural network trained to map an input $x$ to a low-dimensional latent representation $z$, then reconstruct $\hat{x}$ from $z$. Trained exclusively on normal data, it learns a compact manifold of normal operating states. When an anomalous input is presented, the network cannot reconstruct it faithfully, producing a high reconstruction error that serves as the anomaly score.

**Architecture (symmetric encoder-decoder):**

```
Input        20
Encoder:     20  ->  64  ->  32  ->  12   (latent space)
Decoder:     12  ->  32  ->  64  ->  20
Output       20
```

Each layer uses ReLU activation in the encoder and decoder, with a linear output layer. The bottleneck dimension of 12 forces the network to discard noise and retain only the dominant structure of normal telemetry.

**Loss function:** Domain-weighted Mean Squared Error. Each feature dimension $j$ is assigned a weight $w_j$ reflecting the operational criticality of its subsystem:

| Subsystem | Weight | Rationale |
|---|---|---|
| RF power, SNR | 1.5 | Mission-critical communications |
| Voltage, Current, Power | 1.3 | Power subsystem integrity |
| Eclipse indicator | 1.2 | Orbital context |
| Temperature, Pressure | 1.0 | Standard housekeeping |
| Gyro, Mag | 0.8 | Less critical for most missions |

The weighted MSE loss is:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \frac{\sum_{j=1}^{d} w_j \,(x_{ij} - \hat{x}_{ij})^2}{\sum_{j=1}^{d} w_j}$$

This forces the autoencoder to prioritize faithful reconstruction of mission-critical channels (communications and power), making it more sensitive to operationally dangerous anomalies while tolerating slightly higher error on less critical attitude sensors.

The same weighted error, computed per-sample at inference time, is used as the anomaly score.

**Training configuration:**

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | $10^{-3}$ |
| Batch size | 128 |

**Anomaly threshold:** After training, the reconstruction error is computed on every sample in the validation set (which contains only normal points). The decision threshold $\tau$ is set to the 95th percentile of this distribution:

$$\tau = \text{Percentile}_{95}\left(\{ \| x_i - \hat{x}_i \|^2 \}_{i \in \text{val}} \right)$$

A test point $x$ is classified as anomalous if and only if $\| x - \hat{x} \|^2 > \tau$.

```python
from src.models.autoencoder import SatelliteAutoencoder
model = SatelliteAutoencoder(input_dim=20, epochs=50)
model.fit(X_train, X_val)
metrics = model.evaluate(X_test, y_test)
```

---

### 4. Evaluation

All models are evaluated on the held-out test set. Labels are binary: 0 (normal), 1 (anomaly). The following metrics are computed:

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{ROC-AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

where $TP$ = true positives, $FP$ = false positives, $FN$ = false negatives, and the ROC-AUC integrates the true positive rate over all possible decision thresholds.

**Results:**

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| Isolation Forest | 0.89 | 0.87 | 0.88 | 0.91 |
| **Autoencoder** | **0.94** | **0.91** | **0.925** | **0.935** |

The autoencoder meets all target thresholds. The Isolation Forest serves as a strong interpretable baseline but falls short on precision and F1, which is expected given that it does not model temporal structure.

---

## Installation and Usage

```bash
pip install -r requirements.txt

streamlit run app.py
```

The Streamlit dashboard provides:
- Real-time visualization of multivariate telemetry channels
- Detected anomaly markers overlaid on the time series
- Reconstruction error distribution and score histograms
- Side-by-side metrics comparison between both models

A synthetic generator (`src.utils.preprocessing.generate_synthetic_dataset`) is provided for development and testing.
