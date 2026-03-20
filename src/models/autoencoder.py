"""
Autoencoder model for satellite anomaly detection (PyTorch).

Architecture (symmetric):
  Encoder: 20 → 64 → 32 → 12  (latent space)
  Decoder: 12 → 32 → 64 → 20

Loss: MSE  L = (1/N) * sum(||x - x_hat||^2)

Anomaly threshold: 95th percentile of reconstruction error on the
validation set. Samples with error above this threshold are anomalous.

Training hyperparameters:
  epochs     = 50
  optimizer  = Adam (lr=1e-3)
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
LATENT_DIM = 12
HIDDEN_DIMS = (64, 32)
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
THRESHOLD_PERCENTILE = 95


class Autoencoder(nn.Module):
    """
    Symmetric Autoencoder with architecture:
      20 → 64 → 32 → 12 → 32 → 64 → 20

    Parameters
    ----------
    input_dim : int
        Number of input features (default 20).
    hidden_dims : tuple of int
        Hidden layer sizes for encoder (default (64, 32)).
    latent_dim : int
        Bottleneck/latent space dimension (default 12).
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
        encoder_layers += [nn.Linear(prev_dim, latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # Sigmoid to keep outputs in [0, 1] (inputs are Min-Max normalised)
        decoder_layers.append(nn.Sigmoid())
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
    threshold_percentile : int
        Percentile of validation reconstruction error used as anomaly
        threshold (default 95).
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
        threshold_percentile: int = THRESHOLD_PERCENTILE,
        device: str = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold_percentile = threshold_percentile
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
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error."""
        self.model.eval()
        with torch.no_grad():
            t = self._to_tensor(X)
            recon = self.model(t)
            errors = ((t - recon) ** 2).mean(dim=1).cpu().numpy()
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

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(X_train)
            self.train_losses_.append(epoch_loss)

            if X_val is not None:
                val_loss = self._reconstruction_error(X_val).mean()
                self.val_losses_.append(float(val_loss))

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
