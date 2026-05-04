"""Project-wide constants and configuration."""

import os
from pathlib import Path


def _in_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False


IN_COLAB = _in_colab()
REPO_ROOT = Path(
    os.environ.get("CV_FACE_ALIGNMENT_REPO_ROOT",
                   Path(__file__).resolve().parents[1])
).expanduser().resolve()


def _default_workspace_dir() -> Path:
    """Pick a persistent workspace for outputs and datasets."""
    if IN_COLAB:
        return Path('/content/drive/MyDrive/cv-face-alignment')
    return REPO_ROOT


PROJECT_DIR = Path(
    os.environ.get("CV_FACE_ALIGNMENT_WORKSPACE_DIR",
                   _default_workspace_dir())
).expanduser().resolve()


def ensure_workspace_dirs() -> None:
    """Create the workspace subdirectories used across notebooks."""
    for directory in (
        PROJECT_DIR / 'data',
        PROJECT_DIR / 'figures',
        PROJECT_DIR / 'checkpoints',
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_workspace_dirs()

# Backward-compatible aliases: older notebooks still refer to DRIVE_PROJECT_DIR.
DRIVE_PROJECT_DIR = str(PROJECT_DIR)
DATA_DIR          = str(PROJECT_DIR / 'data')
FIGURES_DIR       = str(PROJECT_DIR / 'figures')
CHECKPOINTS_DIR   = str(PROJECT_DIR / 'checkpoints')

TRAIN_NPZ = str(Path(DATA_DIR) / 'face_alignment_training_images.npz')
TEST_NPZ  = str(Path(DATA_DIR) / 'face_alignment_test_images.npz')
SPLIT_NPZ = str(Path(DATA_DIR) / 'train_val_split.npz')

# --- Dataset constants (verified empirically in 01_eda.ipynb) ---
IMAGE_SIZE     = 256
N_LANDMARKS    = 5
N_TRAIN        = 2811
N_TEST         = 554

# Landmark identities (image-coordinate convention; subject's R/L is mirrored)
LANDMARK_NAMES = ('right_eye', 'left_eye', 'nose_tip',
                  'right_mouth', 'left_mouth')

# Index permutation for horizontal flip augmentation:
# eyes swap (0<->1), nose stays (2), mouth corners swap (3<->4)
FLIP_INDICES = (1, 0, 2, 4, 3)

# Per-channel mean/std for image normalisation, normalised to [0, 1]
# (verified empirically in 01_eda.ipynb)
NORMALISE_MEAN = (0.469, 0.374, 0.325)
NORMALISE_STD  = (0.305, 0.268, 0.261)

# --- Reproducibility ---
RANDOM_SEED = 42
VAL_FRACTION = 0.15
