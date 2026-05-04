"""File I/O — including the assignment-provided save_as_csv."""

from pathlib import Path
import numpy as np


def save_as_csv(points: np.ndarray, location: str = '.') -> None:
    """
    Save the points out as a .csv file.

    This function is provided by the assignment brief and must be used as-is
    for the final test submission.

    :param points: numpy array of shape (no_test_images, no_points, 2)
    :param location: directory to save results.csv in. Default: current dir.
    """
    assert points.shape[0] == 554, (
        'wrong number of image points, should be 554 test images'
    )
    assert np.prod(points.shape[1:]) == 5 * 2, (
        'wrong number of points provided. There should be 5 points '
        'with 2 values (x,y) per point'
    )
    np.savetxt(
        location + '/results_task2.csv',
        np.reshape(points, (points.shape[0], -1)),
        delimiter=',',
    )
