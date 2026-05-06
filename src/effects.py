"""Graphical effects applied via face segmentation masks.

Apply a "Take On Me" stylisation as seen in the 1985 Aha
music video — a high-contrast pencil-sketch look with limited grayscale
palette, applied to the face region only.
"""

import numpy as np
import cv2

from . import config


def _posterise(image_gray: np.ndarray, levels: int = 4) -> np.ndarray:
    """Reduce a grayscale image to `levels` discrete intensity bands.

    Mimics charcoal hatching's limited tonal range.
    """
    # Map 0–255 to 0–(levels-1), back to 0–255 with even spacing
    quantised = (image_gray.astype(np.float32) // (256 / levels)).clip(0, levels - 1)
    palette = np.linspace(0, 255, levels, dtype=np.float32)
    return palette[quantised.astype(np.int32)].astype(np.uint8)


def _pencil_edges(image_bgr: np.ndarray,
                   blur_sigma: float = 1.4,
                   canny_low: int = 30,
                   canny_high: int = 60) -> np.ndarray:
    """Extract dark line work using Canny edge detection.

    :return: (H, W) uint8 image, white where there's no edge, dark where
             there's an edge — i.e. ready to multiply against an underlying image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    # Dilate slightly so lines have weight (single-pixel edges look thin)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel)
    # Invert: black ink on white paper
    return 255 - edges


def take_on_me(image: np.ndarray,
                mask: np.ndarray,
                posterise_levels: int = 4,
                edge_blur: float = 1.6,
                canny_low: int = 20,
                canny_high: int = 60,
                tint: tuple = (250, 240, 220)) -> np.ndarray:
    """Apply Take On Me type pencil-sketch stylisation to face pixels.

    Pipeline:
        1. Convert face region to grayscale and posterise
        2. Detect edges via Canny on a Gaussian-blurred grayscale
        3. Multiply edges into the posterised image to add lines to posterized image
        4. Apply a subtle warm tint (paper colour)
        5. Composite over the original via the soft mask

    :param image: (H, W, 3) RGB uint8 image.
    :param mask: (H, W) uint8 mask, 0–255. Soft (feathered) edges blend best.
    :param posterise_levels: number of discrete grey bands.
    :param edge_blur: Gaussian blur sigma applied before Canny.
    :param canny_low/canny_high: Canny hysteresis thresholds.
    :param tint: BGR-style RGB triple to multiply paper colour by; Set to (255, 255, 255) for pure grayscale.
    :return: (H, W, 3) uint8 stylised image.
    """
    H, W = image.shape[:2]

    # 1. Posterise
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    poster = _posterise(gray, levels=posterise_levels)

    # 2. Edges
    edges_inv = _pencil_edges(image, blur_sigma=edge_blur,
                               canny_low=canny_low, canny_high=canny_high)

    # 3. Combine: multiply (in [0,1]) so dark edges darken the posterised image
    poster_f = poster.astype(np.float32) / 255.0
    edges_f  = edges_inv.astype(np.float32) / 255.0
    combined = (poster_f * edges_f * 255.0).clip(0, 255).astype(np.uint8)

    # 4. Tint to a paper colour and broadcast to 3 channels
    tint_arr = np.array(tint, dtype=np.float32) / 255.0
    sketch_rgb = (combined[..., None].astype(np.float32) * tint_arr).clip(0, 255).astype(np.uint8)

    # 5. Composite over the original image using the mask
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    output = (sketch_rgb.astype(np.float32) * alpha
              + image.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    return output