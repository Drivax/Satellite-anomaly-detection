"""
Autoencoder model for satellite anomaly detection (PyTorch).

Architecture (symmetric):
  Encoder: input_dim → 128 → 64 → 32 → latent_dim
  Decoder: latent_dim → 32 → 64 → 128 → input_dim

Loss: Domain-weighted MSE
  L = (1/N) * sum_i sum_j [ w_j * (x_ij - x̂_ij)² ] / sum(w)
  where w_j reflects sensor criticality
    RF power, SNR    : 1.5
    Voltage, Current : 1.3
    Eclipse          : 1.2
    Temperature      : 1.0
    Gyro, Mag        : 0.8

Anomaly threshold: 95th percentile of reconstruction error on the
validation set. Samples with error above this threshold are anomalous.

Training hyperparameters:
    epochs     = 120
    optimizer  = Adam (lr=5e-4, weight_decay=1e-5)
    batch_size = 128
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from pathlib import Path


# Default hyperparameters
INPUT_DIM = 20
LATENT_DIM = 8
HIDDEN_DIMS = (128, 64, 32)
EPOCHS = 120
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
THRESHOLD_PERCENTILE = 95
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15


class Autoencoder(nn.Module):
    """
    Symmetric Autoencoder with architecture:
      20 → 128 → 64 → 32 → 8 → 32 → 64 → 128 → 20

    Parameters
    ----------
    input_dim : int
        Number of input features (default 20).
    hidden_dims : tuple of int
        Hidden layer sizes for encoder (default (128, 64, 32)).
    latent_dim : int
        Bottleneck/latent space dimension (default 8).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: tuple = HIDDEN_DIMS,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))  # no activation on latent
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SatelliteAutoencoder:
    """
    High-level wrapper for training, thresholding and evaluating the
    Autoencoder model.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : tuple of int
        Encoder hidden layer sizes.
    latent_dim : int
        Bottleneck dimension.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Adam learning rate.
    weight_decay : float
        Adam weight decay for mild regularisation.
    threshold_percentile : int
        Percentile of validation reconstruction error used as anomaly
        threshold (default 95).
    patience : int
        Early stopping patience measured in epochs without validation
        improvement.
    feature_weights : np.ndarray or None
        Per-feature weight vector for domain-weighted MSE loss.
        When None, all features are weighted equally (standard MSE).
    device : str or None
        PyTorch device ('cuda', 'cpu').  Auto-detected if None.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: tuple = HIDDEN_DIMS,
        latent_dim: int = LATENT_DIM,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        threshold_percentile: int = THRESHOLD_PERCENTILE,
        patience: int = EARLY_STOPPING_PATIENCE,
        feature_weights: np.ndarray = None,
        device: str = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold_percentile = threshold_percentile
        self.patience = patience
        self.threshold_ = None
        self.train_losses_ = []
        self.val_losses_ = []

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = Autoencoder(input_dim, hidden_dims, latent_dim).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )

        # Domain-weighted loss: per-feature weights for sensor criticality.
        # When the input is a flattened window (window_size * n_features),
        # pass the base per-feature weights and set window_size so they
        # are tiled automatically.
        if feature_weights is not None:
            w = torch.tensor(feature_weights, dtype=torch.float32)
            # Tile weights across the window if input_dim is a multiple
            n_base = len(feature_weights)
            if input_dim > n_base and input_dim % n_base == 0:
                w = w.repeat(input_dim // n_base)
            w = (w / w.sum()).to(self.device)
            self.feature_weights_ = w
        else:
            self.feature_weights_ = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def _weighted_mse(self, x: torch.Tensor, recon: torch.Tensor
                      ) -> torch.Tensor:
        """Compute domain-weighted MSE loss over a batch."""
        sq = (x - recon) ** 2                        # (batch, features)
        if self.feature_weights_ is not None:
            sq = sq * self.feature_weights_           # broadcast weights
        return sq.mean()

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample (weighted) MSE reconstruction error."""
        self.model.eval()
        with torch.no_grad():
            t = self._to_tensor(X)
            recon = self.model(t)
            sq = (t - recon) ** 2
            if self.feature_weights_ is not None:
                sq = sq * self.feature_weights_
            errors = sq.mean(dim=1).cpu().numpy()
        return errors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray = None,
    ) -> "SatelliteAutoencoder":
        """
        Train the autoencoder on normal (or unlabelled) data.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            Training features (Min-Max normalised).
        X_val : np.ndarray, optional
            Validation features for loss tracking and threshold calibration.

        Returns
        -------
        self
        """
        dataset = TensorDataset(self._to_tensor(X_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self._weighted_mse(batch, recon)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(X_train)
            self.train_losses_.append(epoch_loss)

            if X_val is not None:
                val_loss = self._reconstruction_error(X_val).mean()
                self.val_losses_.append(float(val_loss))
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = float(val_loss)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Calibrate anomaly threshold from validation set
        if X_val is not None:
            val_errors = self._reconstruction_error(X_val)
            self.threshold_ = float(
                np.percentile(val_errors, self.threshold_percentile)
            )

        return self

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-sample reconstruction error (higher = more anomalous).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        errors : np.ndarray of shape (n_samples,)
        """
        return self._reconstruction_error(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels using the calibrated threshold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray  (0 = normal, 1 = anomaly)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet (threshold not set).
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "Model has not been fitted or threshold has not been set. "
                "Call fit() with a validation set first."
            )
        errors = self._reconstruction_error(X)
        return (errors > self.threshold_).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Compute Precision, Recall, F1-score and ROC-AUC.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y_true : array-like of shape (n_samples,)

        Returns
        -------
        metrics : dict
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
        """Save model weights and threshold to disk."""
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold_,
                "train_losses": self.train_losses_,
                "val_losses": self.val_losses_,
            },
            path,
        )

    def load(self, path: str) -> "SatelliteAutoencoder":
        """Load model weights and threshold from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.threshold_ = checkpoint.get("threshold")
        self.train_losses_ = checkpoint.get("train_losses", [])
        self.val_losses_ = checkpoint.get("val_losses", [])
        return self
