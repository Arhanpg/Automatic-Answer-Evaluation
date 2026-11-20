"""
backend/red_ink_extractor.py -- Stage 1 Deterministic Preprocessing and Teacher Red-Ink Separation.

Optimized for unconstrained handwritten exam booklets under the 4 GB VRAM RTX 3050 constraint:
- 300 DPI rendering capped at max_width = 2048 via cv2.INTER_CUBIC.
- Dual-band HSV Red-Ink isolation [0, 10] U [170, 180] preventing red marks from occluding handwriting.
- Table gridline suppression prior to cv2.minAreaRect deskewing (MIN_FOREGROUND_PIXELS = 500).
- Controlled Non-Local Means Denoising (h = 10, template 7, search 21).
- Sauvola adaptive local binarization (window_size = 25, k = 0.2) and clean deskewed RGB.
"""
import os
import cv2
import numpy as np
import pymupdf
from PIL import Image
from skimage.filters import threshold_sauvola
from typing import Tuple, Dict, Any, Optional

MIN_FOREGROUND_PIXELS = 500
MAX_WIDTH = 2048

def render_page_dpi(pdf_path: str, page_idx: int, dpi: int = 300, max_width: int = MAX_WIDTH) -> np.ndarray:
    """Renders a PDF page to a BGR numpy array at specified DPI, capping max_width with INTER_CUBIC."""
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # Convert RGB pixmap to BGR numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_h = max(1, int(h * scale))
            bgr = cv2.resize(bgr, (max_width, new_h), interpolation=cv2.INTER_CUBIC)
        return bgr
    finally:
        doc.close()

def extract_and_remove_red_ink(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Separates red examiner marks from student handwriting using dual-band HSV isolation.
    Returns:
      student_clean_bgr: Student handwriting with red teacher marks inpainted to paper white.
      teacher_red_bgr: White background containing only the teacher's red marks (in black/gray).
      red_pixel_fraction: Fraction of pixels identified as red ink.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Dual-band red mask: low red [0, 10] and high red [170, 180]
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixel_fraction = float(np.mean(red_mask > 0))

    if red_pixel_fraction < 0.0001:
        # Virtually no red ink
        white_bg = np.full_like(bgr, 255)
        return bgr.copy(), white_bg, 0.0

    # Dilate red mask to cover anti-aliased red pen margins
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_red = cv2.dilate(red_mask, kernel, iterations=1)

    # Inpaint/replace student layer red pixels with local background
    student_clean = bgr.copy()
    student_clean[dilated_red > 0] = [255, 255, 255]

    # Teacher marks layer: white page with red marks darkened for clarity
    teacher_layer = np.full_like(bgr, 255)
    teacher_layer[dilated_red > 0] = [30, 30, 30]

    return student_clean, teacher_layer, red_pixel_fraction

def preprocess_handwritten_page(bgr: np.ndarray, remove_red: bool = True, suppress_grid: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Complete 4-stage preprocessing pipeline:
      1. Dual-band HSV Red-Ink Extraction & Inpainting (removes examiner annotations).
      2. Table gridline detection & suppression.
      3. OpenCV 4.5+ angle-corrected MinAreaRect deskewing.
      4. Controlled Non-Local Means (NLM) denoising (h=10) & Sauvola adaptive binarization (w=25, k=0.2).
    Returns:
      clean_rgb: Deskewed, red-ink-removed, denoised RGB image (for PaddleOCR, TrOCR, VLMs).
      sauvola_bin: Clean binarized image (for Tesseract and layout line detection).
      skew_angle: Estimated deskew angle in degrees.
    """
    # Stage 1: Dual-band HSV Red-Ink Isolation
    working_bgr = bgr.copy()
    red_fraction = 0.0
    if remove_red:
        working_bgr, _, red_fraction = extract_and_remove_red_ink(working_bgr)

    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Stage 2: Table gridline suppression
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
    grid_mask = cv2.bitwise_or(h_lines, v_lines)

    if suppress_grid:
        # Inpaint gridlines to white in working image
        working_bgr[grid_mask < 200] = [255, 255, 255]
        gray[grid_mask < 200] = 255

    # Stage 3: MinAreaRect deskew angle estimation
    gray_for_skew = gray.copy()
    _, otsu_inv = cv2.threshold(gray_for_skew, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(otsu_inv > 0))

    skew_angle = 0.0
    if len(coords) >= MIN_FOREGROUND_PIXELS:
        rect = cv2.minAreaRect(coords)
        raw_angle = rect[-1]
        w_box, h_box = rect[1]
        # Robust OpenCV 4.5+ minAreaRect angle convention
        if w_box < h_box:
            skew_angle = float(raw_angle)
        else:
            skew_angle = -float(90.0 - raw_angle)

    deskewed_bgr = working_bgr.copy()
    if 0.1 < abs(skew_angle) <= 45.0:
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -skew_angle, 1.0)
        deskewed_bgr = cv2.warpAffine(working_bgr, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Stage 4: Controlled Non-Local Means Denoising on grayscale (h=10 protects minus signs and decimals)
    deskewed_gray = cv2.cvtColor(deskewed_bgr, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.fastNlMeansDenoising(deskewed_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Sauvola local adaptive binarization
    thresh_sauvola = threshold_sauvola(denoised_gray, window_size=25, k=0.2)
    sauvola_bin = ((denoised_gray > thresh_sauvola) * 255).astype(np.uint8)

    # RGB version with denoising applied across channels for OCR
    denoised_bgr = cv2.fastNlMeansDenoisingColored(deskewed_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    clean_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    return clean_rgb, sauvola_bin, skew_angle

def full_preprocess_pipeline(
    bgr: np.ndarray,
    remove_red: bool = True,
    suppress_grid: bool = True,
    deskew: bool = True,
    denoise: bool = True,
    binarize: bool = True
) -> Dict[str, Any]:
    """
    Modular preprocessing executor for reproducible ablation studies.
    Returns:
      dict with keys: 'clean_rgb', 'bin_image', 'skew_angle', 'red_pixel_fraction'
    """
    working_bgr = bgr.copy()
    red_fraction = 0.0
    if remove_red:
        working_bgr, _, red_fraction = extract_and_remove_red_ink(working_bgr)

    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if suppress_grid:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
        grid_mask = cv2.bitwise_or(h_lines, v_lines)
        working_bgr[grid_mask < 200] = [255, 255, 255]
        gray[grid_mask < 200] = 255

    skew_angle = 0.0
    if deskew:
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(otsu_inv > 0))
        if len(coords) >= MIN_FOREGROUND_PIXELS:
            rect = cv2.minAreaRect(coords)
            raw_angle = rect[-1]
            w_box, h_box = rect[1]
            if w_box < h_box:
                skew_angle = float(raw_angle)
            else:
                skew_angle = -float(90.0 - raw_angle)

        if 0.1 < abs(skew_angle) <= 45.0:
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, -skew_angle, 1.0)
            working_bgr = cv2.warpAffine(working_bgr, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
    if denoise:
        denoised_gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised_gray = gray

    if binarize:
        thresh_sauvola = threshold_sauvola(denoised_gray, window_size=25, k=0.2)
        bin_image = ((denoised_gray > thresh_sauvola) * 255).astype(np.uint8)
    else:
        _, bin_image = cv2.threshold(denoised_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    clean_rgb = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB)
    return {
        "clean_rgb": clean_rgb,
        "bin_image": bin_image,
        "skew_angle": skew_angle,
        "red_pixel_fraction": red_fraction
    }
