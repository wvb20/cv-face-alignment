"""Face segmentation — polygon masks from predicted landmarks.

Uses cv2.fillPoly (suggested by the assignment brief) on a polygon
constructed from the 5 predicted facial landmarks plus extrapolated
forehead and chin points.
"""

from typing import Tuple
import numpy as np
import cv2

from . import config


def face_polygon_points(landmarks: np.ndarray,
                         forehead_factor: float = 0.6,
                         chin_factor: float = 0.5,
                         cheek_factor: float = 0.25) -> np.ndarray:
    """
    Builds a 7-point polygon outline of the face from 5 landmarks.

    Landmark order (config.LANDMARK_NAMES):
        0: right_eye, 1: left_eye, 2: nose_tip, 3: right_mouth, 4: left_mouth

    The 5 landmarks alone don't bound a face but they can be used to guide
    where to mask. We extrapolate four extra points to get a polygon roughly
    matching the face outline:
        - forehead-right and forehead-left (above and outside the eyes)
        - right-cheek and left-cheek (outside the eyes, at nose height)
        - chin-bottom (below the mouth midpoint)

    From typical human face morphology there are two "rules" used to derive
    these points (Rule of Thirds for vertical placement, Rule of Fifths for
    horizontal). These ratios are reasonable defaults in the absence of more
    data and can be tuned via the `*_factor` arguments.

    The resulting 7-point polygon traces (counter-clockwise from forehead):
        forehead-right -> forehead-left -> left-cheek -> left-mouth
        -> chin-bottom -> right-mouth -> right-cheek -> (back to forehead-right)

    :param landmarks: (5, 2) array of (x, y) predicted landmark coordinates.
    :param forehead_factor: vertical placement of the forehead points,
        as a fraction of the eye-to-mouth distance. Default 0.6.
    :param chin_factor: vertical placement of the chin-bottom,
        as a fraction of the eye-to-mouth distance. Default 0.5.
    :param cheek_factor: horizontal extension of the cheek and forehead
        points, as a fraction of the inter-ocular distance. Default 0.25.
    :return: (7, 2) polygon vertex array as float64.
    """
    p = np.asarray(landmarks, dtype=np.float64)
    right_eye, left_eye, nose, right_mouth, left_mouth = p

    # Reference scales
    eye_midpoint = (right_eye + left_eye) / 2.0
    mouth_midpoint = (right_mouth + left_mouth) / 2.0
    eye_to_mouth = np.linalg.norm(mouth_midpoint - eye_midpoint)
    iod = np.linalg.norm(left_eye - right_eye)

    # Down-direction in image coordinates (eyes are above mouth, so this
    # vector points downward in pixel terms). Up is its negation.
    down = mouth_midpoint - eye_midpoint
    if np.linalg.norm(down) < 1e-6:
        down = np.array([0.0, 1.0])
    down /= np.linalg.norm(down)
    up = -down

    # Vertical reference y-coordinates
    forehead_y = (eye_midpoint + up * (forehead_factor * eye_to_mouth))[1]
    chin_midpoint = mouth_midpoint + down * (chin_factor * eye_to_mouth)
    chin_y = chin_midpoint[1]
    

    # The distance from nose to eye midpoint is roughly one-third of the face height
    nose_to_eye_dist = np.linalg.norm(eye_midpoint - nose)

    # Define polygon vertices
    forehead_right = np.array([right_eye[0] - cheek_factor * iod, forehead_y])
    hairline_point = eye_midpoint + (up * 1.3 * nose_to_eye_dist) # Usually, the forehead height is roughly equal to the eye-to-nose distance
    forehead_left  = np.array([left_eye[0]  + cheek_factor * iod, forehead_y])
    right_cheek    = np.array([right_eye[0] - cheek_factor * iod, nose[1]])
    left_cheek     = np.array([left_eye[0]  + cheek_factor * iod, nose[1]])
    left_chin      = np.array([left_mouth[0] + cheek_factor * iod, chin_y])
    chin_bottom    = chin_midpoint - (up * nose_to_eye_dist)
    right_chin     = np.array([right_mouth[0] - cheek_factor * iod, chin_y])

    polygon = np.array([
        forehead_right,
        hairline_point,
        forehead_left,
        left_cheek,
        left_chin,
        chin_bottom,
        right_chin,
        right_cheek,
    ], dtype=np.float64)

    return polygon

def face_polygon_mask(image_shape: Tuple[int, int],
                       landmarks: np.ndarray,
                       **polygon_kwargs) -> np.ndarray:
    """
    Build a binary face mask from landmarks using cv2.fillPoly.

    :param image_shape: (H, W) of the image to mask.
    :param landmarks: (5, 2) array of (x, y) landmark coordinates.
    :param polygon_kwargs: passed to face_polygon_points (forehead_factor etc.)
    :return: (H, W) uint8 mask, 255 inside the face polygon, 0 outside.
    """
    H, W = image_shape[:2]
    polygon = face_polygon_points(landmarks, **polygon_kwargs)
    polygon_int = polygon.round().astype(np.int32)

    # cv2.fillPoly expects a list of (N, 1, 2) int arrays
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_int.reshape(-1, 1, 2)], 255)
    return mask


def feathered_face_mask(image_shape: Tuple[int, int],
                         landmarks: np.ndarray,
                         feather_pixels: int = 15,
                         **polygon_kwargs) -> np.ndarray:
    """
    Same as face_polygon_mask but with a blurred (feathered) boundary.

    Useful for blending stylised effects without hard edges. We dilate
    slightly to recover edges lost from a tight polygon, then blur to
    smooth the boundary.

    :param feather_pixels: Gaussian blur sigma in pixels for the soft edge.
    :return: (H, W) uint8 mask, 0–255 with smooth transitions.
    """
    binary = face_polygon_mask(image_shape, landmarks, **polygon_kwargs)
    if feather_pixels > 0:
        # Odd kernel size required by cv2.GaussianBlur; sigma scales with size.
        ksize = max(3, 2 * (feather_pixels // 2) + 1)
        binary = cv2.GaussianBlur(binary, (ksize, ksize), feather_pixels)
    return binary