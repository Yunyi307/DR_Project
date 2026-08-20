"""Fundus-image preprocessing.

Three stages, applied in order:

  1. ``crop_to_roi``     - remove the black border and tightly crop the circular
                           retina, so the informative region fills the frame.
  2. ``ben_graham``      - Ben Graham's colour-normalisation (winner of the 2015
                           Kaggle DR competition): subtract a heavily blurred copy
                           to flatten uneven illumination and boost local detail.
  3. ``apply_clahe``     - Contrast Limited Adaptive Histogram Equalisation on the
                           luminance channel, improving visibility of small lesions
                           (microaneurysms, haemorrhages).

The public entry point ``preprocess_fundus`` composes these and returns an RGB
uint8 image ready for resizing/augmentation. Each stage is exposed separately so
the report/demo can show the effect of each step.
"""
from __future__ import annotations

import cv2
import numpy as np


def crop_to_roi(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop away the uninformative black border around the circular retina.

    A pixel is considered "background" if its grayscale value is below ``tol``.
    Rows/columns that are entirely background are removed. Falls back to the
    original image if the mask is empty (e.g. a fully dark frame).
    """
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray > tol
    if not mask.any():
        return img

    coords = np.ix_(mask.any(1), mask.any(0))
    if img.ndim == 2:
        return img[coords]
    return img[coords[0], coords[1], :]


def ben_graham(img: np.ndarray, sigma_scale: float = 10.0) -> np.ndarray:
    """Ben Graham colour normalisation.

    ``out = 4*img - 4*GaussianBlur(img) + 128``. The blur kernel scales with the
    image size so the effect is resolution independent.
    """
    img = img.astype(np.float32)
    sigma = max(img.shape[:2]) / sigma_scale
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img, 4.0, blurred, -4.0, 128.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE on the L channel of LAB colour space (preserves colour)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def circular_mask(img: np.ndarray) -> np.ndarray:
    """Zero out the corners outside the inscribed circle of the (square-ish) crop.

    Applied after cropping to suppress border artefacts introduced by Ben Graham.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def preprocess_fundus(
    img: np.ndarray,
    use_ben_graham: bool = True,
    use_clahe: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline. Input and output are RGB uint8.

    Order: crop -> (Ben Graham) -> (CLAHE) -> circular mask.
    """
    out = crop_to_roi(img)

    if use_ben_graham:
        out = ben_graham(out)
    if use_clahe:
        out = apply_clahe(out)
    out = circular_mask(out)
    return out

def render_cache_image(img: np.ndarray, cache_size: int = 512,
                       variant: str = "full") -> np.ndarray:
    """Reproduce exactly the offline cache pipeline (scripts/cache_preprocess.py).

    Order: crop -> resize(cache_size) -> [CLAHE -> Ben Graham if 'full'] -> mask.
    Kept identical so that external/inference images see the same distribution the
    model was trained on. Input/output are RGB uint8.
    """
    out = crop_to_roi(img)
    out = cv2.resize(out, (cache_size, cache_size), interpolation=cv2.INTER_AREA)
    if variant == "full":
        out = apply_clahe(out)
        out = ben_graham(out)
    out = circular_mask(out)
    return out


def load_and_preprocess(path: str, size: int | None = None, **kwargs) -> np.ndarray:
    """Read an image file (BGR via OpenCV), preprocess, optionally resize.

    Returns an RGB uint8 array. Raises ``FileNotFoundError`` on unreadable paths.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = preprocess_fundus(rgb, **kwargs)
    if size is not None:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return out
