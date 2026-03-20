"""
Isolation Forest baseline model for satellite anomaly detection.

Anomaly score: s(x, n) = 2^{-E(h(x)) / c(n)}
  - s ≈ 1  → anomaly
  - s ≈ 0  → normal

Hyperparameters (matching the problem specification):
  n_estimators  = 200
  contamination = 0.038   (real anomaly proportion in OPSSAT-AD)
  max_samples   = 0.8
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# Default hyperparameters
N_ESTIMATORS = 200
CONTAMINATION = 0.038
MAX_SAMPLES = 0.8


class SatelliteIsolationForest:
    """
    Wrapper around scikit-learn's IsolationForest for satellite telemetry
    anomaly detection.

    Parameters
    ----------
    n_estimators : int
        Number of base estimators (trees).
    contamination : float
        Expected proportion of anomalies in the training set.
    max_samples : float or int
        Sub-sample size for each tree.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        contamination: float = CONTAMINATION,
        max_samples: float = MAX_SAMPLES,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self.threshold_ = None

    def fit(self, X_train: np.ndarray) -> "SatelliteIsolationForest":
        """
        Fit the Isolation Forest on (anomaly-free) training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.

        Returns
        -------
        self
        """
        self.model.fit(X_train)
        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores in [0, 1] where higher = more anomalous.

        IsolationForest.decision_function returns negative scores where lower
        is more anomalous.  We negate and shift to [0, 1].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly score for each sample (higher = more anomalous).
        """
        raw = self.model.decision_function(X)
        # Negate: IsolationForest convention is lower = more anomalous
        scores = -raw
        # Normalise to [0, 1] for interpretability
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels (0 = normal, 1 = anomaly).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        raw = self.model.predict(X)
        # sklearn returns 1 for normal, -1 for anomaly → convert to 0/1
        return np.where(raw == -1, 1, 0)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Compute Precision, Recall, F1-score and ROC-AUC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_true : array-like of shape (n_samples,)
            Ground-truth labels (0=normal, 1=anomaly).

        Returns
        -------
        metrics : dict
            Dictionary with keys 'precision', 'recall', 'f1', 'roc_auc'.
        """
        y_pred = self.predict(X)
        scores = self.anomaly_scores(X)

        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, scores),
        }
        return metrics

    def save(self, path: str) -> None:
        """Persist the trained model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "SatelliteIsolationForest":
        """Load a previously saved model from disk."""
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.threshold_ = None
        return obj
