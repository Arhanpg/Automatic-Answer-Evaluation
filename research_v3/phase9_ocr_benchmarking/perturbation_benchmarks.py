"""
research_v3/phase9_ocr_benchmarking/perturbation_benchmarks.py

Controlled synthetic perturbation and stress tests for handwriting recognition:
1. Geometric Rotation: +/- 5 degrees, +/- 10 degrees, 180 degrees (inversion)
2. Resolution Downsampling: 300 DPI equivalent -> 150 DPI -> 72 DPI
3. Gaussian Blur / Optical Defocus: sigma = 1.0, 2.0
Evaluates on a representative stratified sample of 50 crops across all 11 subquestions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE9_DIR = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking"
RESULTS_DIR = PHASE9_DIR / "results"
OUTPUT_CSV = RESULTS_DIR / "perturbation_summary.csv"

CROPS_DIR = REPO_ROOT / "research_v3" / "crops"
CROPS_MANIFEST = CROPS_DIR / "crops_manifest.csv"
ADJ_FILE = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "annotations_store" / "adjudications.json"

sys.path.insert(0, str(PHASE9_DIR))
from metrics import evaluate_prediction
from ocr_engines import TesseractEngine


def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    """Rotates image around its center with white background fill."""
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(255, 255, 255))


def downsample_image(image: Image.Image, scale: float) -> Image.Image:
    """Downsamples image by scale factor, then scales back to simulate low-DPI capture."""
    w, h = image.size
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small = image.resize((small_w, small_h), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.NEAREST)


def blur_image(image: Image.Image, sigma: float) -> Image.Image:
    """Applies Gaussian blur to simulate optical defocus."""
    arr = np.array(image.convert("RGB"))
    k_size = int(2 * round(sigma * 2) + 1)
    blurred = cv2.GaussianBlur(arr, (k_size, k_size), sigma)
    return Image.fromarray(blurred, mode="RGB")


def run_perturbations():
    print("=" * 80)
    print("PHASE 9: CONTROLLED SYNTHETIC PERTURBATION STRESS TESTS (50 STRATIFIED CROPS)")
    print("=" * 80)

    import json
    crops_df = pd.read_csv(CROPS_MANIFEST)
    crops_df = crops_df[crops_df["status"] != "omitted_by_student"]
    adjudications = json.loads(ADJ_FILE.read_text(encoding="utf-8"))

    # Select stratified sample of 50 crops across subquestions
    sample_indices = []
    for sq, group in crops_df.groupby("subquestion_id"):
        sample_indices.extend(group.sample(min(len(group), 5), random_state=42).index.tolist())
    sample_df = crops_df.loc[sample_indices[:50]]
    print(f"Sampled {len(sample_df)} stratified answer crops.")

    engine = TesseractEngine()

    perturbation_specs = {
        "clean_baseline": lambda im: im,
        "rot_plus_5deg": lambda im: rotate_image(im, 5.0),
        "rot_minus_5deg": lambda im: rotate_image(im, -5.0),
        "rot_plus_10deg": lambda im: rotate_image(im, 10.0),
        "rot_minus_10deg": lambda im: rotate_image(im, -10.0),
        "rot_180deg_inverted": lambda im: rotate_image(im, 180.0),
        "downsample_50pct": lambda im: downsample_image(im, 0.5),
        "downsample_25pct": lambda im: downsample_image(im, 0.25),
        "gaussian_blur_sigma1": lambda im: blur_image(im, 1.0),
        "gaussian_blur_sigma2": lambda im: blur_image(im, 2.0),
    }

    perturbation_records = []

    for idx, row in sample_df.iterrows():
        sid = row["student_id"]
        sqid = row["subquestion_id"]
        key = f"{sid}_{sqid}"
        gold_text = adjudications.get(key, {}).get("final_adjudicated_transcript", "")

        crop_path = CROPS_DIR / row["crop_filename"]
        if not crop_path.exists():
            continue

        base_img = Image.open(crop_path).convert("RGB")

        for p_name, p_fn in perturbation_specs.items():
            perturbed_img = p_fn(base_img)
            pred = engine.predict(perturbed_img)
            metrics = evaluate_prediction(gold_text, pred["text"], is_omitted=False)

            perturbation_records.append({
                "student_id": sid,
                "subquestion_id": sqid,
                "perturbation": p_name,
                "cer": metrics["cer"],
                "wer": metrics["wer"],
                "math_f1": metrics["math_f1"],
                "num_f1": metrics["num_f1"],
                "char_length": len(pred["text"])
            })

    pdf = pd.DataFrame(perturbation_records)

    # Compute perturbation summary table
    summary_rows = []
    baseline_cer = pdf[pdf["perturbation"] == "clean_baseline"]["cer"].mean()

    for p_name, group in pdf.groupby("perturbation"):
        mean_c = group["cer"].mean()
        mean_w = group["wer"].mean()
        summary_rows.append({
            "perturbation": p_name,
            "sample_count": len(group),
            "macro_cer": round(mean_c, 4),
            "macro_wer": round(mean_w, 4),
            "cer_delta_vs_baseline": round(mean_c - baseline_cer, 4),
            "mean_math_f1": round(group["math_f1"].mean(), 4),
            "mean_num_f1": round(group["num_f1"].mean(), 4),
            "mean_char_count": round(group["char_length"].mean(), 1)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPerturbation summary saved to: {OUTPUT_CSV}")

    print("\n" + "=" * 80)
    print("PERTURBATION STRESS TEST LEADERBOARD (TESSERACT):")
    print("=" * 80)
    print(summary_df.sort_values("cer_delta_vs_baseline").to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_perturbations()
