"""Models — classical Ridge regressors and (later) the CNN."""

from typing import List, Optional
import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.nn.functional as F


from . import config, features

# =============================================================================
# Traditional SIFT based model with cascaded regression 
# =============================================================================


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
    


# =============================================================================
# Deep Learning — direct landmark regression CNN
# =============================================================================


class LandmarkCNN(nn.Module):
    """
    Small convolutional network for direct landmark regression.

    Predicts 5 landmarks × 2 coordinates = 10 outputs per image, in the
    normalised range [-1, 1] (relative to image centre, scaled by half-image).

    Architecture: 4 conv blocks (Conv-BN-ReLU-MaxPool) reducing 256→16,
    global average pool, two FC layers with dropout. ~500K parameters.

    Designed to be compact enough to train in <30 min on a Colab T4
    while having sufficient capacity for ~3K training images.
    """

    def __init__(self, n_landmarks: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_landmarks = n_landmarks
        out_dim = n_landmarks * 2

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3,   32),    # 256 -> 128
            block(32,  64),    # 128 -> 64
            block(64,  128),   # 64 -> 32
            block(128, 256),   # 32 -> 16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 16x16 -> 1x1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: (B, 3, H, W) float in roughly [-2, 2] (post-normalisation)
        :return: (B, n_landmarks, 2) predictions in normalised [-1, 1] coords
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x.view(-1, self.n_landmarks, 2)


def points_to_normalised(points: np.ndarray, image_size: int) -> np.ndarray:
    """
    Convert pixel coordinates to normalised [-1, 1] (centred on image).

    Image centre = (size/2, size/2) maps to (0, 0); corners to (±1, ±1).
    This makes the regression target scale-free and well-behaved for L1/L2 loss.

    :param points: (..., 2) pixel coordinates.
    :param image_size: side length in pixels (assumed square).
    :return: same shape as input, normalised.
    """
    return (points - image_size / 2) / (image_size / 2)


def points_to_pixels(norm_points: np.ndarray, image_size: int) -> np.ndarray:
    """Inverse of points_to_normalised."""
    return norm_points * (image_size / 2) + image_size / 2