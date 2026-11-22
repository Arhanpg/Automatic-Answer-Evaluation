"""
research_v3/phase9_ocr_benchmarking/run_benchmark.py

Executes isolated handwriting recognition benchmarks across genuine OCR engines:
1. Tesseract (LSTM handwritten configuration)
2. PaddleOCR (PP-OCRv6 English recognition with textline orientation)
3. TrOCR (microsoft/trocr-small-handwritten on CUDA)

Evaluates:
- Raw RGB vs Preprocessed (composite illumination + CLAHE + bilateral filtering)
- All 407 subquestion evaluations (392 written crops + 15 verified omissions)
- Standard metrics: CER, WER, Exact Match, Math F1, Numeric F1, Latency, Failure Rate
- Saves comprehensive record-level CSV and engine summary CSV.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE9_DIR = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking"
RESULTS_DIR = PHASE9_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CROPS_DIR = REPO_ROOT / "research_v3" / "crops"
CROPS_MANIFEST = CROPS_DIR / "crops_manifest.csv"
ADJ_FILE = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "annotations_store" / "adjudications.json"

# Import Phase 9 modules
sys.path.insert(0, str(PHASE9_DIR))
from preprocessing import preprocess_raw, preprocess_composite
from metrics import evaluate_prediction
from ocr_engines import TesseractEngine, PaddleOCREngine, TrOCREngine


def run_benchmark():
    print("=" * 80)
    print("PHASE 9: ISOLATED MULTI-MODEL OCR BENCHMARKING & PREPROCESSING ABLATION")
    print("=" * 80)

    if not CROPS_MANIFEST.exists():
        print(f"Error: crops_manifest.csv missing at {CROPS_MANIFEST}")
        sys.exit(1)

    if not ADJ_FILE.exists():
        print(f"Error: adjudications.json missing at {ADJ_FILE}")
        sys.exit(1)

    crops_df = pd.read_csv(CROPS_MANIFEST)
    adjudications = json.loads(ADJ_FILE.read_text(encoding="utf-8"))

    print(f"Loaded {len(crops_df)} records from crops manifest.")
    print(f"Loaded {len(adjudications)} reference records from adjudications.")

    # Initialize engines
    print("\nInitializing OCR Engines...")
    tess_engine = TesseractEngine()
    print("  [1/3] Tesseract Engine initialized.")
    paddle_engine = PaddleOCREngine()
    print("  [2/3] PaddleOCR PP-OCRv6 Engine initialized.")
    trocr_engine = TrOCREngine()
    print("  [3/3] TrOCR-Small Engine initialized on CUDA.")

    engines = {
        "tesseract": tess_engine,
        "paddleocr": paddle_engine,
        "trocr": trocr_engine,
    }

    conditions = {
        "raw": preprocess_raw,
        "preprocessed": preprocess_composite,
    }

    record_results = []
    predictions_log = []

    total_tasks = len(crops_df) * len(engines) * len(conditions)
    print(f"\nBeginning execution of {total_tasks} isolated recognition evaluations...")
    start_time = time.time()

    completed = 0

    for idx, row in crops_df.iterrows():
        sid = row["student_id"]
        sqid = row["subquestion_id"]
        key = f"{sid}_{sqid}"
        is_omitted = str(row["status"]).startswith("omitted")

        ref_meta = adjudications.get(key, {})
        gold_transcript = ref_meta.get("final_adjudicated_transcript", "")

        crop_file = CROPS_DIR / str(row["crop_filename"])

        # Load image if written
        raw_img = None
        if not is_omitted:
            if crop_file.exists():
                try:
                    raw_img = Image.open(crop_file).convert("RGB")
                except Exception as e:
                    print(f"Error loading {crop_file}: {e}")
            else:
                print(f"Warning: Missing crop file on disk: {crop_file}")

        for cond_name, prep_fn in conditions.items():
            # Prepare image
            if is_omitted or raw_img is None:
                proc_img = None
            else:
                proc_img = prep_fn(raw_img)

            for eng_name, engine in engines.items():
                completed += 1

                if is_omitted:
                    hyp_text = ""
                    latency_ms = 0.0
                    success = True
                    err = None
                elif proc_img is None:
                    hyp_text = ""
                    latency_ms = 0.0
                    success = False
                    err = "Crop image unavailable"
                else:
                    pred_res = engine.predict(proc_img)
                    hyp_text = pred_res["text"]
                    latency_ms = pred_res["latency_ms"]
                    success = pred_res["success"]
                    err = pred_res["error"]

                # Evaluate metrics
                metrics = evaluate_prediction(gold_transcript, hyp_text, is_omitted)

                rec = {
                    "student_id": sid,
                    "subquestion_id": sqid,
                    "parent_question_id": f"Q{sqid[0]}",
                    "engine": eng_name,
                    "condition": cond_name,
                    "is_omitted": is_omitted,
                    "latency_ms": latency_ms,
                    "success": success,
                    "error": err,
                    **metrics
                }
                record_results.append(rec)

                predictions_log.append({
                    "student_id": sid,
                    "subquestion_id": sqid,
                    "engine": eng_name,
                    "condition": cond_name,
                    "reference": gold_transcript,
                    "hypothesis": hyp_text,
                    "is_omitted": is_omitted,
                    "cer": metrics["cer"],
                    "wer": metrics["wer"],
                    "exact_match": metrics["exact_match"]
                })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(crops_df):
            elapsed = time.time() - start_time
            rate = completed / max(1e-3, elapsed)
            print(f"  Progress: {idx + 1}/{len(crops_df)} booklets processed ({completed}/{total_tasks} evals, {rate:.1f} evals/sec)")

    # Save results to CSV and JSONL
    results_df = pd.DataFrame(record_results)
    metrics_csv = RESULTS_DIR / "ocr_metrics_by_answer.csv"
    results_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved record-level metrics: {metrics_csv}")

    pred_jsonl = RESULTS_DIR / "predictions.jsonl"
    with open(pred_jsonl, "w", encoding="utf-8") as f:
        for p in predictions_log:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved prediction records: {pred_jsonl}")

    # Generate Engine Summary Table
    # Filter to written items for CER/WER to reflect actual handwriting recognition accuracy
    written_df = results_df[~results_df["is_omitted"]]

    summary_rows = []
    for (eng, cond), group in results_df.groupby(["engine", "condition"]):
        w_group = group[~group["is_omitted"]]
        o_group = group[group["is_omitted"]]

        summary_rows.append({
            "engine": eng,
            "condition": cond,
            "total_records": len(group),
            "written_records": len(w_group),
            "omitted_records": len(o_group),
            "macro_cer": round(w_group["cer"].mean(), 4),
            "macro_wer": round(w_group["wer"].mean(), 4),
            "exact_match_rate": round(w_group["exact_match"].mean(), 4),
            "math_f1": round(w_group["math_f1"].mean(), 4),
            "numeric_f1": round(w_group["num_f1"].mean(), 4),
            "omission_detection_accuracy": round(o_group["omission_correct"].mean(), 4) if len(o_group) > 0 else 1.0,
            "mean_latency_ms": round(w_group["latency_ms"].mean(), 2),
            "success_rate": round(group["success"].mean(), 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "ocr_metrics_by_engine.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved engine summary table: {summary_csv}")

    print("\n" + "=" * 80)
    print("PHASE 9 OCR BENCHMARK SUMMARY LEADERBOARD (WRITTEN HANDWRITING CROPS):")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
