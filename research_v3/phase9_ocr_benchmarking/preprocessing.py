"""
research_v3/phase9_ocr_benchmarking/preprocessing.py

Comprehensive image preprocessing module for Phase 9 handwriting recognition ablation:
1. Raw RGB (unmodified control)
2. Grayscale conversion
3. Global Otsu binarization
4. Local Sauvola adaptive binarization
5. Contrast Limited Adaptive Histogram Equalization (CLAHE)
6. Illumination normalization via morphological background subtraction
7. Composite enhancement pipeline (illumination correction + CLAHE + unsharp mask)
"""

from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def to_cv2_image(image: Image.Image) -> np.ndarray:
    """Converts PIL Image to RGB NumPy array for OpenCV."""
    return np.array(image.convert("RGB"))


def to_pil_image(arr: np.ndarray) -> Image.Image:
    """Converts NumPy array back to PIL Image."""
    if len(arr.shape) == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def preprocess_raw(image: Image.Image) -> Image.Image:
    """Control condition: raw unaltered image."""
    return image.copy()


def preprocess_grayscale(image: Image.Image) -> Image.Image:
    """Converts image to single-channel luminance grayscale."""
    arr = to_cv2_image(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return to_pil_image(gray)


def preprocess_otsu(image: Image.Image) -> Image.Image:
    """Applies Gaussian smoothing followed by global Otsu binarization."""
    arr = to_cv2_image(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return to_pil_image(binary)


def preprocess_sauvola(image: Image.Image, window_size: int = 25, k: float = 0.2, r: float = 128.0) -> Image.Image:
    """Applies local Sauvola adaptive thresholding designed for degraded document images."""
    arr = to_cv2_image(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Local mean and standard deviation via box filter
    mean = cv2.boxFilter(gray, -1, (window_size, window_size))
    sq_mean = cv2.boxFilter(gray * gray, -1, (window_size, window_size))
    variance = np.maximum(0, sq_mean - mean * mean)
    std = np.sqrt(variance)

    # Sauvola threshold: T = m * (1 + k * (s / R - 1))
    threshold = mean * (1.0 + k * (std / r - 1.0))
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return to_pil_image(binary)


def preprocess_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    """Applies Contrast Limited Adaptive Histogram Equalization in LAB color space."""
    arr = to_cv2_image(image)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return to_pil_image(enhanced)


def preprocess_illumination_correction(image: Image.Image) -> Image.Image:
    """Removes non-uniform shadows and page lighting via morphological opening background division."""
    arr = to_cv2_image(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Morphological background estimation with large structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Background division to normalize illumination
    normalized = cv2.divide(gray, background, scale=255)
    return to_pil_image(normalized)


def preprocess_composite(image: Image.Image) -> Image.Image:
    """
    Standardized multi-stage handwriting enhancement pipeline:
    1. Illumination normalization to remove scan gradients
    2. Grayscale CLAHE contrast amplification
    3. Mild bilateral edge-preserving smoothing
    """
    arr = to_cv2_image(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # 1. Illumination normalization
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    norm = cv2.divide(gray, bg, scale=255)

    # 2. CLAHE on normalized image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(norm)

    # 3. Bilateral filter to reduce paper grain while preserving sharp pen strokes
    denoised = cv2.bilateralFilter(contrast, d=5, sigmaColor=30, sigmaSpace=30)
    return to_pil_image(denoised)


PREPROCESSING_REGISTRY = {
    "raw": preprocess_raw,
    "grayscale": preprocess_grayscale,
    "otsu": preprocess_otsu,
    "sauvola": preprocess_sauvola,
    "clahe": preprocess_clahe,
    "illumination": preprocess_illumination_correction,
    "composite": preprocess_composite,
}
