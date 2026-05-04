
import cv2
import numpy as np

from . import config


# Initialize the SIFT detector once at module level.  
# avoids the cost of recreating it for every image).
_SIFT = cv2.SIFT_create()


def compute_sift_at_points(image: np.ndarray,
                            points: np.ndarray,
                            patch_size: int = 16) -> np.ndarray:
    """
    Compute a SIFT descriptor at each given (x, y) location.

    Used for the classical regression baseline (cascaded Ridge on SIFT
    features). Note: this computes descriptors at supplied locations rather than using keypoint detection.

    Adapts the lab's sample panorama SIFT pattern:
        sift = cv2.xfeatures2d.SIFT_create()
        gray = np.uint8(np.mean(image, axis=-1))
        kp, des = sift.detectAndCompute(gray, None)
    To use sift.compute() with hand-built KeyPoints instead, since for
    face alignment we already know where we want descriptors.

    :param image: HxWx3 uint8 image (or HxW grayscale).
    :param points: (M, 2) array of (x, y) sample locations in pixels.
    :param patch_size: SIFT patch diameter, passed to KeyPoint.size.
                       Larger = more context per descriptor, less local detail.
    :return: flat array of shape (128 * M,) — descriptors concatenated
             in the same order as the input points.
    """
    # Convert to grayscale (lecture pattern: mean across colour channels)
    if image.ndim == 3:
        gray = np.uint8(np.mean(image, axis=-1))
    else:
        gray = image.astype(np.uint8)

    # Build a KeyPoint at each requested (x, y) location.
    # cv2.KeyPoint takes (x, y, size) — size is the patch diameter in pixels.
    keypoints = [cv2.KeyPoint(float(x), float(y), float(patch_size))
                 for (x, y) in points]

    # Compute descriptors at the supplied keypoints (no detection).
    # Returns: (filtered_keypoints, descriptors). OpenCV may drop keypoints
    # whose patches fall outside the image — we handle that below.
    kps_out, descriptors = _SIFT.compute(gray, keypoints)

    if descriptors is None or len(kps_out) != len(points):
        # Fallback: pad with zeros for any dropped keypoints.
        # (Rare in practice for face landmarks, since they sit well inside
        # the 256x256 frame, but we'd rather degrade than crash.)
        padded = np.zeros((len(points), 128), dtype=np.float32)
        if descriptors is not None:
            # Map returned descriptors back to their original index by location
            for i, kp in enumerate(kps_out):
                # Find the closest original point; in practice the order
                # is preserved unless points were dropped.
                pt = np.array([kp.pt[0], kp.pt[1]])
                dists = np.linalg.norm(points - pt, axis=1)
                padded[int(np.argmin(dists))] = descriptors[i]
        descriptors = padded

    # Flatten to a single feature vector for the regressor:
    # (M, 128) -> (128 * M,)
    return descriptors.flatten().astype(np.float32)


def mean_shape(train_points: np.ndarray) -> np.ndarray:
    """
    Compute the mean landmark configuration — the cascaded regressor's
    initial guess (p_0).

    :param train_points: (N, 5, 2) training landmarks.
    :return: (5, 2) mean landmark positions.
    """
    return train_points.mean(axis=0)
