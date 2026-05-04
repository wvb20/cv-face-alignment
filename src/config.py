"""Project-wide constants and configuration."""

from pathlib import Path

# --- Paths (relative to project root in Colab) ---
DRIVE_PROJECT_DIR = '/content/drive/MyDrive/cv-face-alignment'
DATA_DIR          = f'{DRIVE_PROJECT_DIR}/data'
FIGURES_DIR       = f'{DRIVE_PROJECT_DIR}/figures'
CHECKPOINTS_DIR   = f'{DRIVE_PROJECT_DIR}/checkpoints'

TRAIN_NPZ = f'{DATA_DIR}/face_alignment_training_images.npz'
TEST_NPZ  = f'{DATA_DIR}/face_alignment_test_images.npz'
SPLIT_NPZ = f'{DATA_DIR}/train_val_split.npz'

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
