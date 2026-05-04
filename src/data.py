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
    import os
    if os.path.exists(config.SPLIT_NPZ):
        split = np.load(config.SPLIT_NPZ)
        return split['train_idx'], split['val_idx']

    rng = np.random.default_rng(seed=config.RANDOM_SEED)
    perm = rng.permutation(config.N_TRAIN)
    n_val = int(config.VAL_FRACTION * config.N_TRAIN)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    np.savez(config.SPLIT_NPZ, train_idx=train_idx, val_idx=val_idx)
    return train_idx, val_idx


# =============================================================================
# PyTorch Dataset with affine augmentation
# =============================================================================

import torch
from torch.utils.data import Dataset


class FaceLandmarksDataset(Dataset):
    """
    PyTorch Dataset yielding (image_tensor, normalised_points) pairs.

    On augmentation:
      - Random rotation ±15°, scale [0.9, 1.1], translation ±10 px,
        applied to BOTH image and landmarks via the same affine matrix.
      - Random horizontal flip with landmark index swap (eyes 0<->1,
        mouth 3<->4, nose 2 stays).
      - Brightness/contrast jitter on the image only.

    All random ops are deterministic given a seed, except dataset-level
    randomness which uses PyTorch's RNG.

    :param images: (N, H, W, 3) uint8.
    :param points: (N, 5, 2) pixel coordinates.
    :param augment: whether to apply augmentation (False for val/test).
    """

    def __init__(self,
                 images: np.ndarray,
                 points: np.ndarray,
                 augment: bool = False) -> None:
        self.images = images
        self.points = points
        self.augment = augment
        self.image_size = config.IMAGE_SIZE

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        """Use PyTorch RNG so DataLoader worker seeds control augmentation."""
        return float(torch.empty((), dtype=torch.float32).uniform_(low, high).item())

    @staticmethod
    def _bernoulli(p: float) -> bool:
        """Sample an event with probability p using the worker-local torch RNG."""
        return bool(torch.rand((), dtype=torch.float32).item() < p)

    def _apply_affine(self, image: np.ndarray, pts: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Random rotation + scale + translation on both image and points."""
        import cv2
        H, W = image.shape[:2]
        cx, cy = W / 2, H / 2

        
        # Initial hyper parameters for training
        #angle = float(np.random.uniform(-15.0, 15.0))
        #scale = float(np.random.uniform(0.9, 1.1))
        #tx    = float(np.random.uniform(-10.0, 10.0))
        #ty    = float(np.random.uniform(-10.0, 10.0))

        # First attempt at adjusting hyper paramters
        angle = self._uniform(-10.0, 10.0)   # was -15, 15
        scale = self._uniform(0.95, 1.05)    # was 0.9, 1.1
        tx    = self._uniform(-5.0, 5.0)     # was -10, 10
        ty    = self._uniform(-5.0, 5.0)     # was -10, 10

        # cv2 builds a 2x3 matrix that maps (x,y,1) -> (x',y')
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        image_aug = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_REFLECT)

        # Apply to points: pts_out = M[:2, :2] @ pts + M[:, 2]
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])  # (5, 3)
        pts_aug = (M @ pts_h.T).T  # (5, 2)
        return image_aug, pts_aug

    def _apply_flip(self, image: np.ndarray, pts: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Horizontal flip; reorders landmarks so they remain consistent."""
        H, W = image.shape[:2]
        image_flip = image[:, ::-1, :].copy()
        pts_flip = pts.copy()
        pts_flip[:, 0] = (W - 1) - pts_flip[:, 0]   # mirror x
        pts_flip = pts_flip[list(config.FLIP_INDICES)]  # swap eyes & mouth
        return image_flip, pts_flip

    def _apply_jitter(self, image: np.ndarray) -> np.ndarray:
        """Brightness ±20% and contrast ±20% in normalised float."""
        f = image.astype(np.float32) / 255.0
        brightness = self._uniform(-0.2, 0.2)
        contrast = self._uniform(0.8, 1.2)
        f = (f - 0.5) * contrast + 0.5 + brightness
        f = np.clip(f, 0, 1)
        return (f * 255).astype(np.uint8)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx].copy()
        pts = self.points[idx].astype(np.float32).copy()

        # Initial Hyperparameter Selection
        #if self.augment:
        #    if np.random.rand() < 0.8:
        #        image, pts = self._apply_affine(image, pts)
        #    if np.random.rand() < 0.5:
        #        image, pts = self._apply_flip(image, pts)
        #    if np.random.rand() < 0.5:
        #        image = self._apply_jitter(image)

        # First attempt at adjusting hyper paramters
        if self.augment:
            if self._bernoulli(0.5):                    # was 0.8
                image, pts = self._apply_affine(image, pts)
            if self._bernoulli(0.5):                    # unchanged
                image, pts = self._apply_flip(image, pts)
            if self._bernoulli(0.3):                    # was 0.5
                image = self._apply_jitter(image)



        # Normalise image: uint8 [0,255] -> float [0,1] -> standardised.
        # Use np.float32 for the constants too — Python floats are 64-bit,
        # which leaks float64 into the output and breaks MSELoss.
        f = image.astype(np.float32) / np.float32(255.0)
        mean = np.array(config.NORMALISE_MEAN, dtype=np.float32)
        std  = np.array(config.NORMALISE_STD,  dtype=np.float32)
        f = (f - mean) / std
        # HWC -> CHW, force contiguous float32
        image_t = torch.from_numpy(np.ascontiguousarray(f.transpose(2, 0, 1),
                                                          dtype=np.float32))

        # Normalise points to [-1, 1] — explicit float32 cast prevents
        # dtype mismatch with the model's float32 output (MSELoss is strict)
        pts_norm = ((pts - self.image_size / 2) / (self.image_size / 2)).astype(np.float32)
        pts_t = torch.from_numpy(pts_norm)

        return image_t, pts_t
