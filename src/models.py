"""Models — classical Ridge regressors and the CNN for landmark regression."""

from typing import List, Optional
import numpy as np
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn

from . import config, features


# =============================================================================
# Traditional SIFT-based model with cascaded regression
# =============================================================================


class CascadedRidgeRegressor:
    """
    Cascaded linear regressor for face alignment (Supervised Descent Method).
    See Lesson 5, Supervised Descent for reference.

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

    def _extract_features(self,
                           images: np.ndarray,
                           shapes: np.ndarray) -> np.ndarray:
        """SIFT descriptors at each image's current shape estimate.

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

    def fit(self,
            images: np.ndarray,
            points: np.ndarray,
            verbose: bool = True) -> 'CascadedRidgeRegressor':
        """Train the cascade. Returns self for chaining."""
        self.mean_shape_ = features.mean_shape(points)
        N = len(images)
        current = np.tile(self.mean_shape_, (N, 1, 1)).astype(np.float64)

        self.regressors = []
        for stage in range(self.n_stages):
            X = self._extract_features(images, current)
            y = (points - current).reshape(N, -1)
            reg = Ridge(alpha=self.alpha)
            reg.fit(X, y)
            self.regressors.append(reg)
            delta = reg.predict(X).reshape(N, config.N_LANDMARKS, 2)
            current = current + delta

            if verbose:
                err = np.linalg.norm(current - points, axis=2).mean()
                print(f'  Stage {stage + 1}/{self.n_stages}: '
                      f'mean per-landmark error = {err:.2f} px')
        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict (N, 5, 2) landmarks for a batch of images."""
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
    Convolutional regressor for direct landmark prediction.

    Predicts 5 landmarks × 2 coordinates = 10 outputs per image, in the
    normalised range [-1, 1] (relative to image centre, scaled by half-image).

    Key design changes / considerations:
      - CoordConv addition: x/y coordinate channels concatenated to RGB so conv
        filters have access to absolute position information. 
      - Previous simple model was collapsing to mean-shape predictions because
        their features are translation-equivariant, meaning that there was very 
        little spatial information of their location in the image being considered
        in the simpler model.
      - GroupNorm not BatchNorm: stable for the small batch sizes we use, otherwise
        again the model was collapsing to mean-shape predictions 
      - Tanh output bounded to [-1, 1] with bias initialised so the
        network starts predicting the dataset mean shape on epoch 0, matching
        outputs to the earlier Cascade model for ease of comparison.
    """

    def __init__(self, n_landmarks: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_landmarks = n_landmarks
        out_dim = n_landmarks * 2

        def norm(channels: int) -> nn.Module:
            return nn.GroupNorm(8, channels)

        def block(in_c, out_c):
            """Single conv + norm + ReLU + maxpool."""
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                norm(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # Input is 5 channels: 3 RGB + 2 coord channels (CoordConv)
        # This was originally just the 3 RGB channels but the lack of
        # Spatial information in the model was resulting in poor predictions
        # that collapsed to the mean-shape.
        self.features = nn.Sequential(
            block(5,   32),    # 256 -> 128
            block(32,  64),    # 128 -> 64
            block(64,  128),   # 64 -> 32
            block(128, 192),   # 32 -> 16
        )

        self.spatial_head = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

        # Bias-init the final linear so the network's prediction
        # is the dataset mean shape — forces it to learn image residuals
        # rather than collapsing to the trivial mean.
        final_linear = self.head[-1]
        mean_norm = np.zeros(out_dim, dtype=np.float32)
        if n_landmarks == 5:
            mean_norm = np.array([
                -0.375, -0.187,    # right eye
                 0.375, -0.195,    # left eye
                 0.008,  0.117,    # nose
                -0.227,  0.383,    # right mouth
                 0.250,  0.375,    # left mouth
            ], dtype=np.float32)
        # Pre-tanh values (network applies tanh in forward)
        mean_pre_tanh = np.arctanh(np.clip(mean_norm, -0.999, 0.999))
        with torch.no_grad():
            final_linear.bias.copy_(torch.from_numpy(mean_pre_tanh))

    @staticmethod
    def _coordinate_channels(x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel x/y channels in [-1, 1] for CoordConv-style input."""
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype)
        yy = yy.view(1, 1, h, 1).expand(b, 1, h, w)
        xx = xx.view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([xx, yy], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: (B, 3, H, W) post-normalisation float tensor.
        :return: (B, n_landmarks, 2) predictions in normalised [-1, 1] coords.
        This outputs to the same format as the cascade model for ease of comparison. 
        """
        # Defensive NaN guard for cv2.warpAffine edge cases
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat([x, self._coordinate_channels(x)], dim=1)
        x = self.features(x)
        x = self.spatial_head(x)
        x = self.head(x)
        x = torch.tanh(x)
        return x.view(-1, self.n_landmarks, 2)


def points_to_normalised(points: np.ndarray, image_size: int) -> np.ndarray:
    """Pixel coordinates -> normalised [-1, 1] (centred on image)."""
    return (points - image_size / 2) / (image_size / 2)


def points_to_pixels(norm_points: np.ndarray, image_size: int) -> np.ndarray:
    """Inverse of points_to_normalised; clips to [-1, 1] for safety."""
    norm_points = np.clip(norm_points, -1.0, 1.0)
    return norm_points * (image_size / 2) + image_size / 2