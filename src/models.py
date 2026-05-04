"""Models — classical Ridge regressors and (later) the CNN."""

from typing import List, Optional
import numpy as np
from sklearn.linear_model import Ridge

from . import config, features


class CascadedRidgeRegressor:
    """
    Cascaded linear regressor for face alignment (Supervised Descent Method).

    Each stage k computes SIFT descriptors at the current shape estimate
    p_{k-1}, predicts an update Δp_k via a Ridge regressor trained on
    residuals, and updates p_k = p_{k-1} + Δp_k.

    Stage 0 starts from the mean shape (computed at fit time).

    Reference: Xiong & De la Torre (2013), "Supervised Descent Method
    and its Applications to Face Alignment".

    :param n_stages: number of cascade stages (typical: 3–5)
    :param alpha: Ridge regularisation (typical: 1.0–100.0)
    :param patch_size: SIFT patch diameter
    """

    def __init__(self,
                 n_stages: int = 4,
                 alpha: float = 10.0,
                 patch_size: int = 16) -> None:
        self.n_stages = n_stages
        self.alpha = alpha
        self.patch_size = patch_size
        self.regressors: List[Ridge] = []
        self.mean_shape_: Optional[np.ndarray] = None  # (5, 2)

    # --- Internals ----------------------------------------------------------

    def _extract_features(self,
                           images: np.ndarray,
                           shapes: np.ndarray) -> np.ndarray:
        """
        SIFT descriptors at each image's current shape estimate.

        :param images: (N, H, W, 3) uint8.
        :param shapes: (N, 5, 2) per-image current landmark estimates.
        :return: (N, 5*128) feature matrix.
        """
        N = len(images)
        feat_dim = config.N_LANDMARKS * 128
        feats = np.zeros((N, feat_dim), dtype=np.float32)
        for i in range(N):
            feats[i] = features.compute_sift_at_points(
                images[i], shapes[i], patch_size=self.patch_size
            )
        return feats

    # --- Public API ---------------------------------------------------------

    def fit(self,
            images: np.ndarray,
            points: np.ndarray,
            verbose: bool = True) -> 'CascadedRidgeRegressor':
        """
        Train the cascade.

        :param images: (N, H, W, 3) training images.
        :param points: (N, 5, 2) ground-truth landmarks.
        :param verbose: print per-stage progress.
        :return: self (for method chaining).
        """
        self.mean_shape_ = features.mean_shape(points)
        N = len(images)

        # Stage 0: start every image from the mean shape
        current = np.tile(self.mean_shape_, (N, 1, 1)).astype(np.float64)

        self.regressors = []
        for stage in range(self.n_stages):
            # Features at the current estimate
            X = self._extract_features(images, current)
            # Target: residual from current to ground truth, flattened
            y = (points - current).reshape(N, -1)

            reg = Ridge(alpha=self.alpha)
            reg.fit(X, y)
            self.regressors.append(reg)

            # Update the current estimate with predicted residual
            delta = reg.predict(X).reshape(N, config.N_LANDMARKS, 2)
            current = current + delta

            if verbose:
                # Mean per-landmark Euclidean error in pixels
                err = np.linalg.norm(current - points, axis=2).mean()
                print(f'  Stage {stage + 1}/{self.n_stages}: '
                      f'mean per-landmark error = {err:.2f} px')

        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Predict landmarks for a batch of images by running the trained cascade.

        :param images: (N, H, W, 3) images.
        :return: (N, 5, 2) predicted landmarks.
        """
        if self.mean_shape_ is None or not self.regressors:
            raise RuntimeError('Model has not been fit yet.')

        N = len(images)
        current = np.tile(self.mean_shape_, (N, 1, 1)).astype(np.float64)

        for reg in self.regressors:
            X = self._extract_features(images, current)
            delta = reg.predict(X).reshape(N, config.N_LANDMARKS, 2)
            current = current + delta

        return current