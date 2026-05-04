"""Classical feature extraction — SIFT descriptors at landmark positions."""

import numpy as np

from . import config


def compute_sift_at_points(image: np.ndarray,
                            points: np.ndarray,
                            patch_size: int = 16) -> np.ndarray:
    """
    Compute a SIFT descriptor at each given (x, y) location.

    Used for the classical regression baseline (cascaded Ridge on SIFT
    features). Note: we are NOT using SIFT keypoint *detection* — we
    compute descriptors at supplied locations, as recommended by the brief.

    :param image: HxW or HxWx3 uint8 image.
    :param points: (M, 2) array of (x, y) sample locations.
    :param patch_size: size of the SIFT patch (passed to KeyPoint.size).
    :return: flat array of shape (128 * M,) — descriptors concatenated.
    """
    raise NotImplementedError


def mean_shape(train_points: np.ndarray) -> np.ndarray:
    """
    Compute the mean landmark configuration — the cascaded regressor's
    initial guess (p_0).

    :param train_points: (N, 5, 2) training landmarks.
    :return: (5, 2) mean landmark positions.
    """
    raise NotImplementedError
