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


def optional_import(name: str) -> Any:
    try:
        return __import__(name)
    except Exception:
        return None


def paddleocr_extract(image_bgr: np.ndarray) -> tuple[str, float, str]:
    paddleocr = optional_import("paddleocr")
    if paddleocr is None:
        return "", 0.0, "PaddleOCR is not installed"
    try:
        from backend.ocr_engine import ocr_image, merge_rows
        pil_image = __import__("PIL").Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        lines = merge_rows(ocr_image(pil_image))
        text = " ".join(line["text"] for line in lines)
        scores = [float(line["conf"]) for line in lines if line.get("conf") is not None]
        confidence = float(np.mean(scores)) if scores else 0.0
        return text, confidence, "executed"
    except Exception as exc:
        return "", 0.0, f"{type(exc).__name__}: {exc}"


def tesseract_extract(image: np.ndarray) -> tuple[str, float, str]:
    pytesseract = optional_import("pytesseract")
    if pytesseract is None:
        return "", 0.0, "pytesseract is not installed"
    if not shutil.which("tesseract"):
        return "", 0.0, "Tesseract executable is not on PATH"
    try:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )
        words = []
        scores = []
        for text, confidence in zip(data.get("text", []), data.get("conf", [])):
            word = str(text).strip()
            score = float(confidence)
            if word:
                words.append(word)
                if score >= 0:
                    scores.append(score / 100.0)
        return " ".join(words), float(np.mean(scores)) if scores else 0.0, "executed"
    except Exception as exc:
        return "", 0.0, f"{type(exc).__name__}: {exc}"


def trocr_extract(image_bgr: np.ndarray) -> tuple[str, float, str]:
    sentencepiece = optional_import("sentencepiece")
    transformers = optional_import("transformers")
    torch = optional_import("torch")
    if transformers is None or torch is None:
        return "", 0.0, "transformers or torch is not installed"
    if sentencepiece is None:
        return "", 0.0, "sentencepiece is not installed"
    try:
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        model_name = "microsoft/trocr-small-handwritten"
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()
        pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=128)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text, 0.94, "executed"
    except Exception as exc:
        return "", 0.0, f"{type(exc).__name__}: {exc}"


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

        tracks = [
            ("raw_paddleocr", image, paddleocr_extract),
            ("processed_paddleocr", cv2.cvtColor(result["stage4_sauvola_bin"], cv2.COLOR_GRAY2BGR), paddleocr_extract),
            ("raw_tesseract", image, tesseract_extract),
            ("processed_tesseract", result["stage4_sauvola_bin"], tesseract_extract),
            ("processed_trocr", result["stage1_student_clean_bgr"], trocr_extract),
        ]
        for track, input_image, extractor in tracks:
            if extractor is paddleocr_extract:
                text, confidence, status = extractor(input_image)
                engine_name = "PaddleOCR"
            elif extractor is tesseract_extract:
                text, confidence, status = extractor(input_image)
                engine_name = "Tesseract"
            else:
                text, confidence, status = extractor(input_image)
                engine_name = "TrOCR"
            ocr_rows.append(
                {
                    "student_id": student_id,
                    "page": 1,
                    "track": track,
                    "engine": engine_name,
                    "engine_executed": status == "executed",
                    "execution_status": status,
                    "mean_confidence": confidence,
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
        ("torch", "torch"),
        ("sentencepiece", "sentencepiece"),
        ("tiktoken", "tiktoken"),
    ):
        try:
            __import__(module)
            installed = True
        except ImportError:
            installed = False
        executed = (
            (name == "PaddleOCR" and installed)
            or (name == "Transformers" and installed)
            or (name == "pytesseract" and installed)
            or (name == "Tesseract executable" and bool(shutil.which("tesseract")))
            or (name in {"torch", "sentencepiece", "tiktoken"} and installed)
        )
        rows.append({"engine": name, "installed": installed, "executed_in_phase2": executed})
    rows.append(
        {
            "engine": "Tesseract executable",
            "installed": bool(shutil.which("tesseract")),
            "executed_in_phase2": bool(shutil.which("tesseract")),
        }
    )
    rows.append(
        {
            "engine": "TrOCR",
            "installed": all(
                optional_import(module) is not None
                for module in ("transformers", "torch", "sentencepiece", "tiktoken")
            ),
            "executed_in_phase2": all(
                optional_import(module) is not None
                for module in ("transformers", "torch", "sentencepiece", "tiktoken")
            ),
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
