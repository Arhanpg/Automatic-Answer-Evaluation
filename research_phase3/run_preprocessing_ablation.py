"""Run a real, reference-free preprocessing ablation for representative pages."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import time
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from skimage.filters import threshold_sauvola

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.independent_pipeline import FourStageImagePreprocessor
from backend.red_ink_extractor import render_page_dpi

DATASET = Path(os.environ.get("ANSWER_EVAL_DATASET", r"F:\Automatic answer evaluation dataset"))
OUT = ROOT / "research_phase3"
SAMPLES = ("s1", "s2", "s18", "s29")


def sha256_array(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def grid_suppression(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
    grid = cv2.bitwise_or(horizontal, vertical)
    result = gray.copy()
    result[grid >= 200] = 255
    return result


def run() -> None:
    if not shutil.which("tesseract"):
        raise RuntimeError("Tesseract executable is required")

    OUT.mkdir(parents=True, exist_ok=True)
    preprocessor = FourStageImagePreprocessor(max_width=2048)
    rows = []

    for student_id in SAMPLES:
        pdf = DATASET / "All PDF's" / f"{student_id}.pdf"
        if not pdf.exists():
            continue
        original = render_page_dpi(str(pdf), 0)
        stage1, teacher, red_fraction = preprocessor.stage1_red_ink_separation(original)
        deskewed, skew = preprocessor.stage2_grayscale_and_deskew(stage1)
        denoised = preprocessor.stage3_denoising(deskewed)
        sauvola = preprocessor.stage4_sauvola_binarization(denoised)

        conditions = {
            "original": cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
            "red_ink_removed": cv2.cvtColor(stage1, cv2.COLOR_BGR2GRAY),
            "red_removed_plus_deskew": deskewed,
            "grid_suppression": grid_suppression(stage1),
            "denoising": denoised,
            "sauvola_binarization": sauvola,
            "complete_existing_pipeline": sauvola,
        }

        for condition, image in conditions.items():
            started = time.perf_counter()
            try:
                text = pytesseract.image_to_string(image, config="--psm 6").strip()
                status = "executed"
            except Exception as exc:
                text = ""
                status = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            rows.append({
                "student_id": student_id,
                "page": 1,
                "condition": condition,
                "engine": "Tesseract",
                "execution_status": status,
                "latency_seconds": round(elapsed, 4),
                "character_count": len(text),
                "word_count": len(text.split()),
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "image_sha256": sha256_array(image),
                "red_pixel_fraction": round(float(red_fraction), 6),
                "skew_angle_deg": round(float(skew), 4),
                "cer": "",
                "wer": "",
                "math_symbol_error": "",
                "reference_alignment": "not computed; answer-region ground truth unresolved",
            })

    fields = list(rows[0])
    with (OUT / "preprocessing_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recorded {len(rows)} real preprocessing/OCR ablation rows")


if __name__ == "__main__":
    run()
