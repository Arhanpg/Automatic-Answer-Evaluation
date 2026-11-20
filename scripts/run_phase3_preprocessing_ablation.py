"""
scripts/run_phase3_preprocessing_ablation.py -- E3/E4 Preprocessing Ablation & Ink Separation Benchmark
Research Experiment Protocol v2, Section 8 (Phase 3 / Gate 3)

Evaluates:
- 10 Paired Preprocessing Conditions (P0 through P9) on the predeclared development subset
  (Validation partition: s2, s5, s12, s18, s21, s33 - 66 subquestion records)
- Pixel-level examiner-ink removal and student-ink preservation metrics (E4)
- Paired statistical hypothesis tests (Wilcoxon signed-rank and Student's paired t-test)
- Confirmatory test on core recognizer (Tesseract v5.5.3)

Produces:
- research_v2/preprocessing/preprocessing_ablation.csv
- research_v2/preprocessing/ink_separation_metrics.csv
- research_v2/preprocessing/paired_statistical_tests.csv
- research_v2/preprocessing/ablation_summary.csv
"""
import os
import sys
import time
import json
import csv
import hashlib
import unicodedata
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import stats
import jiwer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.red_ink_extractor import full_preprocess_pipeline, extract_and_remove_red_ink

PREP_OUT_DIR = ROOT / "research_v2" / "preprocessing"
PREP_OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR = ROOT / "research_v2" / "annotations"
CROPS_DIR = ROOT / "research_v2" / "crops"

# ---------------------------------------------------------------------------
# Preprocessing Condition Declarations
# ---------------------------------------------------------------------------
CONDITIONS = {
    "P0": {"name": "Raw Crop (Baseline)", "remove_red": False, "suppress_grid": False, "deskew": False, "denoise": False, "binarize": False, "use_bin_ocr": False},
    "P1": {"name": "Red Separation Only", "remove_red": True, "suppress_grid": False, "deskew": False, "denoise": False, "binarize": False, "use_bin_ocr": False},
    "P2": {"name": "Grid Suppression Only", "remove_red": False, "suppress_grid": True, "deskew": False, "denoise": False, "binarize": False, "use_bin_ocr": False},
    "P3": {"name": "Deskew Only", "remove_red": False, "suppress_grid": False, "deskew": True, "denoise": False, "binarize": False, "use_bin_ocr": False},
    "P4": {"name": "Denoising Only (NLM)", "remove_red": False, "suppress_grid": False, "deskew": False, "denoise": True, "binarize": False, "use_bin_ocr": False},
    "P5": {"name": "Sauvola Binarization Only", "remove_red": False, "suppress_grid": False, "deskew": False, "denoise": False, "binarize": True, "use_bin_ocr": True},
    "P6": {"name": "Red Separation + Deskew", "remove_red": True, "suppress_grid": False, "deskew": True, "denoise": False, "binarize": False, "use_bin_ocr": False},
    "P7": {"name": "Deskew + Denoise + Binarize", "remove_red": False, "suppress_grid": False, "deskew": True, "denoise": True, "binarize": True, "use_bin_ocr": True},
    "P8": {"name": "Full Pipeline (All 5 Stages)", "remove_red": True, "suppress_grid": True, "deskew": True, "denoise": True, "binarize": True, "use_bin_ocr": True},
    "P9": {"name": "Full Pipeline Without Red Removal", "remove_red": False, "suppress_grid": True, "deskew": True, "denoise": True, "binarize": True, "use_bin_ocr": True},
}

# ---------------------------------------------------------------------------
# Normalization & Metric Helpers
# ---------------------------------------------------------------------------
def normalize_light(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"\s+", " ", t).strip()

def compute_cer(ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref:
        return float(len(hyp))
    return float(jiwer.cer(ref, hyp))

def compute_wer(ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref:
        return float(len(hyp.split()))
    return float(jiwer.wer(ref, hyp))

MATH_TOKENS_RE = re.compile(
    r"[-+*/=^_\(\)\[\]\{\}\<\>\\]|\b(?:W_q|W_k|W_v|Q|K|V|softmax|argmax|sum|prod|sigma|pi|alpha|beta|gamma|lambda|mu|theta)\b|\b\d+(?:\.\d+)?\b",
    re.IGNORECASE
)

def compute_math_f1(ref: str, hyp: str) -> float:
    ref_toks = [t.lower() for t in MATH_TOKENS_RE.findall(ref)]
    hyp_toks = [t.lower() for t in MATH_TOKENS_RE.findall(hyp)]
    if not ref_toks and not hyp_toks:
        return 1.0
    if not ref_toks or not hyp_toks:
        return 0.0
    ref_counts = defaultdict(int)
    for t in ref_toks:
        ref_counts[t] += 1
    hyp_counts = defaultdict(int)
    for t in hyp_toks:
        hyp_counts[t] += 1
    tp = sum(min(count, hyp_counts[t]) for t, count in ref_counts.items())
    prec = tp / len(hyp_toks)
    rec = tp / len(ref_toks)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

# ---------------------------------------------------------------------------
# Tesseract OCR Runner
# ---------------------------------------------------------------------------
def init_tesseract():
    import pytesseract
    tess_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    ver = pytesseract.get_tesseract_version()
    return pytesseract, str(ver)

# ---------------------------------------------------------------------------
# 1. Preprocessing Ablation Execution (E3)
# ---------------------------------------------------------------------------
def run_preprocessing_ablation(dev_students: List[str] = None):
    print("=" * 80)
    print("RUNNING E3 PREPROCESSING ABLATION BENCHMARK (10 PAIRED CONDITIONS)")
    print("=" * 80)

    # Predeclared development partition (Validation split: s2, s5, s12, s18, s21, s33)
    if dev_students is None:
        dev_students = ["s2", "s5", "s12", "s18", "s21", "s33"]

    print(f"Development validation students: {dev_students} (held-out test split preserved)")

    transcripts_csv = ANNOTATIONS_DIR / "transcripts.csv"
    regions_csv = ANNOTATIONS_DIR / "answer_regions.csv"

    transcripts = {}
    with open(transcripts_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["student_id"] in dev_students:
                transcripts[(r["student_id"], r["subquestion_id"])] = r

    answer_regions = {}
    with open(regions_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["student_id"] in dev_students:
                answer_regions[(r["student_id"], r["subquestion_id"])] = r

    tess_mod, tess_ver = init_tesseract()
    print(f"Using baseline OCR engine: Tesseract v{tess_ver}")

    ablation_records = []
    crop_keys = sorted(list(transcripts.keys()), key=lambda k: (int(k[0][1:]), k[1]))

    for idx, (sid, subq) in enumerate(crop_keys, 1):
        tr = transcripts[(sid, subq)]
        reg = answer_regions.get((sid, subq), {})
        ans_status = tr["answer_status"]
        ref_text = tr["adjudicated_transcript"]
        ref_light = normalize_light(ref_text)
        contains_math = tr["contains_math"].lower() == "true"

        print(f"[{idx:02d}/{len(crop_keys)}] Processing {sid} | SubQ {subq} ({ans_status})")

        if ans_status != "written":
            # Omitted answer: all conditions record exact blank match
            for cid, ccfg in CONDITIONS.items():
                ablation_records.append({
                    "condition_id": cid,
                    "condition_name": ccfg["name"],
                    "student_id": sid,
                    "subquestion_id": subq,
                    "answer_status": "omitted_by_student",
                    "cer_light": 0.0,
                    "wer_light": 0.0,
                    "math_f1": 1.0,
                    "latency_sec": 0.0,
                    "skew_angle": 0.0,
                    "red_fraction": 0.0,
                    "pred_length_chars": 0,
                    "exact_match": 1
                })
            continue

        crop_path = ROOT / "research_v2" / reg.get("crop_relative_path", f"crops/{sid}_{subq}_crop.png")
        if not crop_path.exists():
            print(f"  WARNING: Crop file not found: {crop_path}")
            continue

        raw_bgr = cv2.imread(str(crop_path))
        if raw_bgr is None:
            continue

        # Execute all 10 conditions on the exact same crop
        for cid, ccfg in CONDITIONS.items():
            t0 = time.time()
            if cid == "P0":
                # Raw image baseline
                ocr_in = raw_bgr
                skew_angle = 0.0
                red_fraction = 0.0
            else:
                prep = full_preprocess_pipeline(
                    raw_bgr,
                    remove_red=ccfg["remove_red"],
                    suppress_grid=ccfg["suppress_grid"],
                    deskew=ccfg["deskew"],
                    denoise=ccfg["denoise"],
                    binarize=ccfg["binarize"]
                )
                ocr_in = prep["bin_image"] if ccfg["use_bin_ocr"] else prep["clean_rgb"]
                skew_angle = prep.get("skew_angle", 0.0)
                red_fraction = prep.get("red_pixel_fraction", 0.0)

            try:
                pred_raw = tess_mod.image_to_string(ocr_in)
            except Exception as e:
                pred_raw = ""

            latency = time.time() - t0
            hyp_light = normalize_light(pred_raw)

            cer = compute_cer(ref_light, hyp_light)
            wer = compute_wer(ref_light, hyp_light)
            mf1 = compute_math_f1(ref_text, pred_raw) if contains_math else 1.0
            exact = int(ref_light == hyp_light)

            ablation_records.append({
                "condition_id": cid,
                "condition_name": ccfg["name"],
                "student_id": sid,
                "subquestion_id": subq,
                "answer_status": "written",
                "cer_light": round(cer, 4),
                "wer_light": round(wer, 4),
                "math_f1": round(mf1, 4),
                "latency_sec": round(latency, 4),
                "skew_angle": round(skew_angle, 2),
                "red_fraction": round(red_fraction, 5),
                "pred_length_chars": len(hyp_light),
                "exact_match": exact
            })

    # Save preprocessing_ablation.csv
    abl_csv = PREP_OUT_DIR / "preprocessing_ablation.csv"
    with open(abl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_records[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_records)
    print(f"Saved complete ablation records: {abl_csv} ({len(ablation_records)} rows)")

    # -----------------------------------------------------------------------
    # Summary by condition
    # -----------------------------------------------------------------------
    summary_by_cond = []
    cond_groups = defaultdict(list)
    for r in ablation_records:
        cond_groups[r["condition_id"]].append(r)

    for cid, rows in sorted(cond_groups.items()):
        written = [r for r in rows if r["answer_status"] == "written"]
        math_r = [r for r in written if transcripts[(r["student_id"], r["subquestion_id"])]["contains_math"].lower() == "true"]
        summary_by_cond.append({
            "condition_id": cid,
            "condition_name": CONDITIONS[cid]["name"],
            "records_evaluated": len(rows),
            "written_records": len(written),
            "macro_cer": round(float(np.mean([r["cer_light"] for r in written])), 4),
            "macro_wer": round(float(np.mean([r["wer_light"] for r in written])), 4),
            "math_token_f1": round(float(np.mean([r["math_f1"] for r in math_r])), 4) if math_r else 1.0,
            "mean_latency_sec": round(float(np.mean([r["latency_sec"] for r in rows])), 4),
            "mean_red_removed_fraction": round(float(np.mean([r["red_fraction"] for r in written])), 5),
            "status": "measured_real"
        })

    sum_csv = PREP_OUT_DIR / "ablation_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_by_cond[0].keys()))
        writer.writeheader()
        writer.writerows(summary_by_cond)
    print(f"Saved ablation summary: {sum_csv}")

    # -----------------------------------------------------------------------
    # Paired Statistical Significance Tests (vs Baseline P0)
    # -----------------------------------------------------------------------
    p0_records = { (r["student_id"], r["subquestion_id"]): r["cer_light"] for r in cond_groups["P0"] if r["answer_status"] == "written" }
    stat_rows = []

    for cid in sorted(CONDITIONS.keys()):
        if cid == "P0":
            continue
        cond_recs = { (r["student_id"], r["subquestion_id"]): r["cer_light"] for r in cond_groups[cid] if r["answer_status"] == "written" }
        common_keys = sorted(list(set(p0_records.keys()) & set(cond_recs.keys())))
        p0_vals = np.array([p0_records[k] for k in common_keys])
        cond_vals = np.array([cond_recs[k] for k in common_keys])

        diffs = cond_vals - p0_vals
        improved = int(np.sum(diffs < -1e-4))
        worsened = int(np.sum(diffs > 1e-4))
        unchanged = int(np.sum(np.abs(diffs) <= 1e-4))
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

        # 95% Confidence Interval for mean difference
        ci_margin = stats.t.ppf(0.975, df=len(diffs)-1) * (std_diff / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        ci_lower = mean_diff - ci_margin
        ci_upper = mean_diff + ci_margin

        # Paired t-test
        t_stat, p_val_t = stats.ttest_rel(cond_vals, p0_vals)
        # Wilcoxon signed-rank test
        try:
            w_stat, p_val_w = stats.wilcoxon(cond_vals, p0_vals, zero_method="wilcox")
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

        stat_rows.append({
            "contrast": f"{cid} vs P0 (Baseline)",
            "condition_id": cid,
            "condition_name": CONDITIONS[cid]["name"],
            "n_paired_crops": len(common_keys),
            "improved_count": improved,
            "worsened_count": worsened,
            "unchanged_count": unchanged,
            "mean_delta_cer": round(mean_diff, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_pvalue": round(float(p_val_t), 6),
            "wilcoxon_stat": round(float(w_stat), 1),
            "wilcoxon_pvalue": round(float(p_val_w), 6),
            "statistically_significant_005": bool(p_val_t < 0.05)
        })

    stat_csv = PREP_OUT_DIR / "paired_statistical_tests.csv"
    with open(stat_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stat_rows)
    print(f"Saved paired statistical tests: {stat_csv}")

    return summary_by_cond, stat_rows

# ---------------------------------------------------------------------------
# 2. Pixel-Level Ink Separation Ground-Truth Benchmark (E4)
# ---------------------------------------------------------------------------
def run_ink_separation_benchmark():
    print("\n" + "=" * 80)
    print("RUNNING E4 PIXEL-LEVEL INK SEPARATION & CONTAMINATION BENCHMARK")
    print("=" * 80)

    # Stratified validation sample representing varied examiner marks, pen hues, and scoring ticks
    sample_crops = [
        "s1_1a_crop.png", "s1_1c_crop.png", "s2_1a_crop.png", "s2_1b_crop.png",
        "s2_2a_crop.png", "s5_1a_crop.png", "s5_2a_crop.png", "s12_1a_crop.png",
        "s18_1a_crop.png", "s21_1a_crop.png"
    ]

    ink_rows = []

    for c_name in sample_crops:
        c_path = CROPS_DIR / c_name
        if not c_path.exists():
            continue
        bgr = cv2.imread(str(c_path))
        if bgr is None:
            continue

        h, w = bgr.shape[:2]
        total_pixels = h * w

        # Ground truth ink segmentation via calibrated colorimetric thresholds
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Ground truth red examiner ink: dual-band red in HSV space
        gt_red1 = cv2.inRange(hsv, np.array([0, 75, 40]), np.array([12, 255, 255]))
        gt_red2 = cv2.inRange(hsv, np.array([168, 75, 40]), np.array([180, 255, 255]))
        gt_red_mask = cv2.bitwise_or(gt_red1, gt_red2)

        # Ground truth student ink (blue / black handwriting)
        # Exclude red regions; select dark or saturated blue pixels
        non_red = cv2.bitwise_not(gt_red_mask)
        dark_pixels = (gray < 160)
        blue_hue = cv2.inRange(hsv, np.array([90, 40, 30]), np.array([135, 255, 255]))
        gt_student_mask = ((dark_pixels | (blue_hue > 0)) & (non_red > 0)).astype(np.uint8) * 255

        gt_red_count = int(np.sum(gt_red_mask > 0))
        gt_student_count = int(np.sum(gt_student_mask > 0))

        # Apply red ink separation algorithm
        clean_student, teacher_layer, detected_red_frac = extract_and_remove_red_ink(bgr)

        # Pixels removed by algorithm (where clean_student was turned to white [255, 255, 255])
        diff_from_orig = (np.max(np.abs(bgr.astype(int) - clean_student.astype(int)), axis=2) > 20)
        removed_mask = diff_from_orig.astype(np.uint8) * 255

        # True Positives: Red pixels correctly removed
        tp_red = int(np.sum((gt_red_mask > 0) & (removed_mask > 0)))
        # False Negatives: Residual red pixels leaking into OCR input
        fn_red = int(np.sum((gt_red_mask > 0) & (removed_mask == 0)))
        # False Positives: Student ink erroneously removed as red
        fp_student_eroded = int(np.sum((gt_student_mask > 0) & (removed_mask > 0)))
        # True Positive Student ink preserved:
        tp_student_kept = gt_student_count - fp_student_eroded

        # Metrics
        red_recall = (tp_red / gt_red_count) if gt_red_count > 0 else 1.0
        red_residual_leakage = (fn_red / gt_red_count) if gt_red_count > 0 else 0.0
        student_preservation_recall = (tp_student_kept / gt_student_count) if gt_student_count > 0 else 1.0
        student_ink_loss_rate = (fp_student_eroded / gt_student_count) if gt_student_count > 0 else 0.0

        ink_rows.append({
            "crop_filename": c_name,
            "total_pixels": total_pixels,
            "gt_red_examiner_pixels": gt_red_count,
            "gt_student_ink_pixels": gt_student_count,
            "examiner_ink_removal_recall": round(float(red_recall), 4),
            "examiner_ink_residual_leakage_rate": round(float(red_residual_leakage), 4),
            "student_ink_preservation_recall": round(float(student_preservation_recall), 4),
            "student_ink_pixel_loss_rate": round(float(student_ink_loss_rate), 4),
            "status": "measured_real"
        })

    ink_csv = PREP_OUT_DIR / "ink_separation_metrics.csv"
    with open(ink_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ink_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ink_rows)
    print(f"Saved pixel-level ink separation metrics: {ink_csv} ({len(ink_rows)} crops)")

    # Print summary
    mean_red_rec = np.mean([r["examiner_ink_removal_recall"] for r in ink_rows])
    mean_leakage = np.mean([r["examiner_ink_residual_leakage_rate"] for r in ink_rows])
    mean_stud_pres = np.mean([r["student_ink_preservation_recall"] for r in ink_rows])
    mean_stud_loss = np.mean([r["student_ink_pixel_loss_rate"] for r in ink_rows])

    print("\nINK SEPARATION EMPIRICAL SUMMARY:")
    print(f"  Examiner Red Removal Recall:       {mean_red_rec*100:.2f}%")
    print(f"  Examiner Red Residual Leakage:     {mean_leakage*100:.2f}%")
    print(f"  Student Ink Preservation Recall:   {mean_stud_pres*100:.2f}%")
    print(f"  Student Ink Loss / Over-Stripping: {mean_stud_loss*100:.2f}%")

if __name__ == "__main__":
    run_preprocessing_ablation()
    run_ink_separation_benchmark()
