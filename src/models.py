"""Models — classical Ridge regressors and the CNN."""

from typing import List, Optional
import numpy as np


class CascadedRidgeRegressor:
    """
    Cascaded linear regressor for face alignment.

    At each cascade stage k, computes SIFT features at the current shape
    estimate p_{k-1}, and predicts an update Δp via a Ridge regressor
    trained on residuals.

    See Xiong & De la Torre (2013), "Supervised Descent Method".
    """

    def __init__(self, n_stages: int = 4, alpha: float = 1.0,
                 patch_size: int = 16) -> None:
        raise NotImplementedError

    def fit(self, images: np.ndarray, points: np.ndarray) -> 'CascadedRidgeRegressor':
        """Train the cascade. Returns self for chaining."""
        raise NotImplementedError

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict (N, 5, 2) landmarks for a batch of images."""
        raise NotImplementedError


# CNN model class will go here in Phase 3 (deep learning baseline).
# Stub kept here so 03_cnn.ipynb has a clear home for the architecture.
