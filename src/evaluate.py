"""Error metrics — NME, CED, and the assignment-provided euclid_dist."""

import numpy as np

from . import config


def euclid_dist(pred_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean distance between pairs of points.

    Provided by the assignment brief — kept as-is.

    :param pred_pts: predicted points, shape (N, 2) or flat (2N,).
    :param gt_pts:   ground truth points, same shape.
    :return: array of shape (no_points,) with per-point distances.
    """
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))


def inter_ocular_distance(points: np.ndarray) -> np.ndarray:
    """
    Compute inter-ocular distance for each face in a batch.

    :param points: (N, 5, 2) ground-truth landmark array.
    :return: (N,) array of distances between landmark 0 and landmark 1.
    """
    raise NotImplementedError


def nme(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Normalised Mean Error per image, normalised by inter-ocular distance.

    NME_i = mean over landmarks of euclidean distance / IOD_i.

    :param pred: (N, 5, 2) predicted landmarks in pixel coordinates.
    :param gt:   (N, 5, 2) ground-truth landmarks.
    :return: (N,) array of per-image NMEs (unitless, typically 0.01–0.20).
    """
    raise NotImplementedError


def cumulative_error_distribution(errors: np.ndarray
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (sorted_errors, cumulative_fractions) for plotting a CED curve.

    :param errors: (N,) array of per-image errors (e.g. NMEs).
    :return: (x, y) arrays for plt.step / plt.plot.
    """
    raise NotImplementedError


def failure_rate(errors: np.ndarray, threshold: float = 0.10) -> float:
    """Fraction of images with error above the threshold (default 10%)."""
    raise NotImplementedError
