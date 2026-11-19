"""Run a non-destructive Phase 2 preprocessing/OCR evidence audit.

Only metadata, hashes, OCR lengths, and aggregate image statistics are written
to the repository. Visual stage images are written to a caller-selected local
directory and are not suitable for publication or upload without authorization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.independent_pipeline import FourStageImagePreprocessor, compute_cer_wer
from backend.red_ink_extractor import render_page_dpi
from backend import evaluation


DATASET_ROOT = Path(os.environ.get("ANSWER_EVAL_DATASET", r"F:\Automatic answer evaluation dataset"))
AUDIT = ROOT / "research_audit"
SAMPLES = ("s1", "s2", "s18", "s29")


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with (AUDIT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_stage_images(result: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    images = {
        "0_original.jpg": result["original_bgr"],
        "1_student_clean.jpg": result["stage1_student_clean_bgr"],
        "1_teacher_marks.jpg": result["stage1_teacher_marks_bgr"],
        "2_deskewed_gray.jpg": result["stage2_deskewed_gray"],
        "3_denoised_gray.jpg": result["stage3_denoised_gray"],
        "4_sauvola_binary.jpg": result["stage4_sauvola_bin"],
    }
    for name, image in images.items():
        cv2.imwrite(str(destination / name), image)


def run_image_audit(output_dir: Path) -> None:
    preprocessor = FourStageImagePreprocessor(max_width=2048)
    rows = []
    ocr_rows = []
    source_dir = DATASET_ROOT / "All PDF's"

    for student_id in SAMPLES:
        pdf_path = source_dir / f"{student_id}.pdf"
        if not pdf_path.exists():
            rows.append({"student_id": student_id, "status": "source PDF missing"})
            continue

        image = render_page_dpi(str(pdf_path), 0)
        result = preprocessor.process_image(image)
        sample_dir = output_dir / student_id / "page_001"
        save_stage_images(result, sample_dir)
        rows.append(
            {
                "student_id": student_id,
                "page": 1,
                "source_path": str(pdf_path.resolve()),
                "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                "render_shape": "x".join(str(v) for v in image.shape),
                "red_pixel_fraction": round(float(result["stage1_red_fraction"]), 6),
                "skew_angle_deg": round(float(result["stage2_skew_angle"]), 4),
                "original_sha256": sha256_array(result["original_bgr"]),
                "student_clean_sha256": sha256_array(result["stage1_student_clean_bgr"]),
                "teacher_marks_sha256": sha256_array(result["stage1_teacher_marks_bgr"]),
                "deskewed_sha256": sha256_array(result["stage2_deskewed_gray"]),
                "denoised_sha256": sha256_array(result["stage3_denoised_gray"]),
                "binarized_sha256": sha256_array(result["stage4_sauvola_bin"]),
                "source_visual_review": "local artifact generated; no independent human claim",
                "visual_output_dir": str(sample_dir.resolve()),
            }
        )

        raw_text = pytesseract.image_to_string(image)
        processed_text = pytesseract.image_to_string(result["stage4_sauvola_bin"])
        for track, text in (("raw_tesseract", raw_text), ("processed_tesseract", processed_text)):
            ocr_rows.append(
                {
                    "student_id": student_id,
                    "page": 1,
                    "track": track,
                    "engine": "Tesseract",
                    "engine_executed": True,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "character_count": len(text.strip()),
                    "word_count": len(text.split()),
                    "reference_alignment": "not computed; repository has no page-to-question transcription mapping",
                    "cer": "",
                    "wer": "",
                    "math_symbol_accuracy": "",
                    "exact_expression_rate": "",
                }
            )

    write_csv("phase2_preprocessing_samples.csv", rows)
    write_csv("phase2_actual_ocr_runs.csv", ocr_rows)


def run_engine_availability() -> None:
    rows = []
    for name, module in (
        ("PaddleOCR", "paddleocr"),
        ("Transformers", "transformers"),
        ("pytesseract", "pytesseract"),
    ):
        try:
            __import__(module)
            installed = True
        except ImportError:
            installed = False
        rows.append({"engine": name, "installed": installed, "executed_in_phase2": name == "pytesseract" and installed})
    rows.append(
        {
            "engine": "Tesseract executable",
            "installed": bool(shutil.which("tesseract")),
            "executed_in_phase2": bool(shutil.which("tesseract")),
        }
    )
    write_csv("phase2_engine_availability.csv", rows)


def run_metric_audit() -> None:
    cases = [
        ("exact", "abc", "abc"),
        ("substitution", "axc", "abc"),
        ("insertion", "abxc", "abc"),
        ("math_minus", "x - 1", "x + 1"),
        ("empty_both", "", ""),
    ]
    rows = []
    for case, hypothesis, reference in cases:
        independent = compute_cer_wer(hypothesis, reference)
        aggregate = evaluation.compute_ocr_metrics([hypothesis], [reference])
        rows.append(
            {
                "case": case,
                "hypothesis": hypothesis,
                "reference": reference,
                "independent_cer": independent["cer"],
                "independent_wer": independent["wer"],
                "aggregate_cer": aggregate["cer"],
                "aggregate_wer": aggregate["wer"],
                "aggregate_math_symbol_accuracy": aggregate["math_symbol_accuracy"],
                "exact_expression_rate": aggregate["exact_expression_rate"],
            }
        )
    write_csv("phase2_metric_audit.csv", rows)


def main() -> None:
    output_dir = Path(
        os.environ.get(
            "PHASE2_VISUAL_OUTPUT_DIR",
            Path(tempfile.gettempdir()) / "automatic-answer-evaluation-phase2-visuals",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_image_audit(output_dir)
    run_engine_availability()
    run_metric_audit()
    print(json.dumps({"metadata_dir": str(AUDIT), "visual_output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
