from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Standardize input features using statistics learned from training data."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()

    def fit(self, features: np.ndarray) -> None:
        self._validate(features)
        self._scaler.fit(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        self._validate(features)

        if not hasattr(self._scaler, "mean_"):
            raise RuntimeError("Scaler must be fitted before transform().")

        return self._scaler.transform(features).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self._validate(features)
        return self._scaler.fit_transform(features).astype(np.float32)

    def _validate(self, features: np.ndarray) -> None:
        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a NumPy array.")

        if features.ndim != 2:
            raise ValueError(
                f"Expected 2D feature matrix, got shape {features.shape}."
            )

        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or infinite values.")

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler