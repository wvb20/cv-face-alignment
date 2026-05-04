"""Plotting helpers — including the assignment-provided visualise_pts."""

import matplotlib.pyplot as plt
import numpy as np


def visualise_pts(img: np.ndarray, pts: np.ndarray) -> None:
    """
    Display an image with its landmark points overlaid.

    Provided by the assignment brief — kept as-is.
    """
    plt.imshow(img)
    plt.plot(pts[:, 0], pts[:, 1], '+g')
    plt.show()


def plot_predictions_grid(images: np.ndarray,
                           preds: np.ndarray,
                           gts: np.ndarray | None = None,
                           n: int = 9,
                           save_path: str | None = None) -> None:
    """
    Plot an n-image grid with predictions (and optional ground truth).

    :param images: (N, H, W, 3) image array.
    :param preds:  (N, 5, 2) predicted landmarks.
    :param gts:    (N, 5, 2) ground-truth landmarks (optional).
    :param n:      number of images to plot (default 9, square grid).
    :param save_path: if given, save the figure to this path.
    """
    raise NotImplementedError
