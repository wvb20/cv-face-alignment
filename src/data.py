"""Data loading, splits, and PyTorch Dataset."""

from typing import Tuple
import numpy as np

from . import config


def load_train() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training images and landmark points.

    :return: (images, points) where
             images is (N, 256, 256, 3) uint8 and
             points is (N, 5, 2) float64 in pixel coordinates.
    """
    with np.load(config.TRAIN_NPZ, allow_pickle=True) as d:
        return d['images'].copy(), d['points'].copy()


def load_test() -> np.ndarray:
    """
    Load the test images (no landmarks).

    :return: images array of shape (554, 256, 256, 3) uint8.
    """
    with np.load(config.TEST_NPZ, allow_pickle=True) as d:
        return d['images'].copy()


def get_split() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx) for the canonical train/val split.

    The split is created on first call (seeded by config.RANDOM_SEED)
    and persisted to config.SPLIT_NPZ. Subsequent calls load it.

    :return: (train_idx, val_idx) — index arrays into the training set.
    """
    raise NotImplementedError('Implemented in 01_eda.ipynb; move it here.')
