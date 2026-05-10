"""Error metrics — NME, CED, and the assignment-provided euclid_dist."""

import numpy as np
from . import config


def euclid_dist(pred_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean distance between pairs of points.

    Provided with the assignment 

    :param pred_pts: predicted points, shape (N, 2) or flat (2N,).
    :param gt_pts:   ground truth points, same shape.
    :return: array of shape (no_points,) with per-point distances.
    """
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))




def inter_ocular_distance(points: np.ndarray) -> np.ndarray:
    """
    Compute distance between eyes for each face in a batch.

    :param points: (N, 5, 2) ground-truth landmark array.
    :return: (N,) array of distances between landmark 0 and landmark 1.
    """
    return np.linalg.norm(points[:, 1, :] - points[:, 0, :], axis=1)


def nme(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Normalised Mean Error per image, normalised by inter-ocular distance (IOD).

    NME_i = mean over landmarks of euclidean distance / IOD_i.

    :param pred: (N, 5, 2) predicted landmarks in pixel coordinates.
    :param gt:   (N, 5, 2) ground-truth landmarks.
    :return: (N,) array of per-image NMEs (unitless, typically 0.01–0.20).
    """
    # Per-image, per-landmark Euclidean distance
    pred = np.nan_to_num(pred, nan=0.0, posinf=config.IMAGE_SIZE, neginf=0.0)
    gt = np.nan_to_num(gt, nan=0.0, posinf=config.IMAGE_SIZE, neginf=0.0)
    per_lm_err = np.linalg.norm(pred - gt, axis=2)            # (N, 5)
    iod = np.clip(inter_ocular_distance(gt), 1e-6, None)      # (N,)
    # Mean over landmarks, normalised by IOD
    errors = per_lm_err.mean(axis=1) / iod                    # (N,)
    return np.nan_to_num(errors, nan=np.inf, posinf=np.inf, neginf=np.inf)


def cumulative_error_distribution(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (sorted_errors, cumulative_fractions) for plotting a CED curve.

    :param errors: (N,) array of per-image errors (e.g. NMEs).
    :return: (x, y) arrays for plt.step / plt.plot.
    """
    sorted_errors = np.sort(errors)
    cumulative = np.linspace(0.0, 1.0, len(errors), endpoint=True)
    return sorted_errors, cumulative


def failure_rate(errors: np.ndarray, threshold: float = 0.10) -> float:
    """Fraction of images with error above the threshold (default 10%)."""
    return float((errors > threshold).mean())
