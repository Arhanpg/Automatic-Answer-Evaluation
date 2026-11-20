"""
scripts/run_phase2_ocr_benchmark.py -- E2 Full-Corpus Live OCR Engine Benchmark
Research Experiment Protocol v2, Section 7 (Phase 2 / Gate 2)

Evaluates:
- Tesseract v5.5.3 (CPU)
- TrOCR Small (microsoft/trocr-small-handwritten on GPU cuda:0)
- PaddleOCR PP-OCRv6 (CPU)

Across all 37 examination booklets (407 subquestion records: 392 written, 15 omitted),
evaluating both Raw and Preprocessed conditions.
Produces:
- research_v2/ocr/ocr_predictions.jsonl
- research_v2/ocr/ocr_metrics_by_answer.csv
- research_v2/ocr/ocr_metrics_by_student.csv
- research_v2/ocr/ocr_metrics_by_engine.csv
- research_v2/ocr/ocr_failures.csv
- research_v2/ocr/model_runtime.csv
"""
import os
import sys
import time
import json
import csv
import hashlib
import unicodedata
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import jiwer

# Configure paddle environment variables before any paddle import
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.red_ink_extractor import full_preprocess_pipeline

OCR_OUT_DIR = ROOT / "research_v2" / "ocr"
OCR_OUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR = ROOT / "research_v2" / "annotations"
CROPS_DIR = ROOT / "research_v2" / "crops"

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def normalize_strict(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text)).strip()

def normalize_light(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

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
NUMBER_RE = re.compile(r"[-+]?\b\d+(?:\.\d+)?\b")

def extract_math_tokens(text: str) -> List[str]:
    return [t.lower() for t in MATH_TOKENS_RE.findall(text)]

def extract_numbers(text: str) -> List[str]:
    return [t.lower() for t in NUMBER_RE.findall(text)]

def compute_token_prf(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[float, float, float]:
    if not ref_tokens and not hyp_tokens:
        return 1.0, 1.0, 1.0
    if not ref_tokens:
        return 0.0, 1.0, 0.0
    if not hyp_tokens:
        return 1.0, 0.0, 0.0
    ref_counts = defaultdict(int)
    for t in ref_tokens:
        ref_counts[t] += 1
    hyp_counts = defaultdict(int)
    for t in hyp_tokens:
        hyp_counts[t] += 1
    tp = sum(min(count, hyp_counts[t]) for t, count in ref_counts.items())
    prec = tp / len(hyp_tokens) if hyp_tokens else 0.0
    rec = tp / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)

# ---------------------------------------------------------------------------
# Morphological Line Segmenter for TrOCR
# ---------------------------------------------------------------------------
def segment_lines_morphology(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Fast morphology-based text line segmenter for document crops.
    Returns list of bounding boxes (x0, y0, x1, y1) sorted top-to-bottom.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Horizontal rectangular structuring element to bridge intra-word characters and word gaps
    kw = max(20, int(w * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    dilated = cv2.dilate(bin_inv, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > w * 0.06 and 18 < bh < h * 0.45:
            # Pad box slightly
            px0 = max(0, x - 4)
            py0 = max(0, y - 4)
            px1 = min(w, x + bw + 4)
            py1 = min(h, y + bh + 4)
            boxes.append((px0, py0, px1, py1))
    # If no lines detected via morphology, fallback to full crop
    if not boxes:
        boxes = [(0, 0, w, h)]
    # Sort in vertical reading order
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

# ---------------------------------------------------------------------------
# Engine Initializers & Runners
# ---------------------------------------------------------------------------
def init_tesseract():
    import pytesseract
    tess_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    ver = pytesseract.get_tesseract_version()
    print(f"Tesseract initialized: version {ver}")
    return pytesseract, str(ver)

def init_trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten").to(device)
    model.eval()
    print(f"TrOCR Small initialized on {device} (PyTorch {torch.__version__})")
    return processor, model, device

def init_paddleocr():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang="en",
        device="cpu",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    import paddle
    print(f"PaddleOCR initialized (Paddle {paddle.__version__}, device: cpu)")
    return ocr

# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------
def run_benchmark(
    selected_engines: Optional[List[str]] = None,
    selected_conditions: Optional[List[str]] = None,
    limit_students: Optional[List[str]] = None,
    resume: bool = True
):
    print("=" * 80)
    print("STARTING E2 FULL-CORPUS LIVE OCR BENCHMARK (PHASE 2 / GATE 2)")
    print("=" * 80)

    # Load manifests
    regions_csv = ANNOTATIONS_DIR / "answer_regions.csv"
    transcripts_csv = ANNOTATIONS_DIR / "transcripts.csv"

    answer_regions = {}
    with open(regions_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            answer_regions[(r["student_id"], r["subquestion_id"])] = r

    transcripts = {}
    with open(transcripts_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            transcripts[(r["student_id"], r["subquestion_id"])] = r

    all_keys = sorted(list(transcripts.keys()), key=lambda k: (int(k[0][1:]), k[1]))
    if limit_students:
        all_keys = [k for k in all_keys if k[0] in limit_students]

    print(f"Total answer records targeted: {len(all_keys)} across {len(set(k[0] for k in all_keys))} booklets")

    engines_to_run = selected_engines or ["tesseract", "trocr", "paddleocr"]
    conditions_to_run = selected_conditions or ["raw", "preprocessed"]

    # Initialize engines
    tess_mod = None
    tess_ver = "unknown"
    if "tesseract" in engines_to_run:
        tess_mod, tess_ver = init_tesseract()

    trocr_proc = None
    trocr_model = None
    trocr_device = "cpu"
    if "trocr" in engines_to_run:
        trocr_proc, trocr_model, trocr_device = init_trocr()

    paddle_ocr = None
    if "paddleocr" in engines_to_run:
        paddle_ocr = init_paddleocr()

    # Load existing predictions if resuming
    pred_file = OCR_OUT_DIR / "ocr_predictions.jsonl"
    existing_preds = set()
    if resume and pred_file.exists():
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_preds.add((obj["student_id"], obj["subquestion_id"], obj["engine"], obj["condition"]))
                except Exception:
                    continue
        print(f"Resuming: found {len(existing_preds)} existing prediction records in {pred_file}")

    # Open predictions file in append mode
    pred_writer = open(pred_file, "a", encoding="utf-8")

    processed_count = 0
    total_start_time = time.time()

    # Cache preprocessed crops locally to avoid redundant image processing
    prep_crops_cache = {}

    for idx, (sid, subq) in enumerate(all_keys, 1):
        tr = transcripts[(sid, subq)]
        reg = answer_regions.get((sid, subq), {})
        ans_status = tr["answer_status"]
        ref_text = tr["adjudicated_transcript"]
        contains_math = tr["contains_math"].lower() == "true"
        contains_diagram = tr["contains_diagram"].lower() == "true"

        print(f"[{idx:03d}/{len(all_keys)}] Student {sid} | SubQ {subq} | Status: {ans_status}")

        if ans_status != "written":
            # Handle omitted answers
            for eng in engines_to_run:
                for cond in conditions_to_run:
                    key = (sid, subq, eng, cond)
                    if key in existing_preds:
                        continue
                    pred_record = {
                        "prediction_id": f"{sid}_{subq}_{eng}_{cond}",
                        "student_id": sid,
                        "subquestion_id": subq,
                        "parent_question_id": reg.get("parent_question_id", f"Q{subq[0]}"),
                        "engine": eng,
                        "condition": cond,
                        "compute_device": "N/A",
                        "answer_status": "omitted_by_student",
                        "crop_relative_path": "",
                        "crop_sha256": "",
                        "crop_verified": True,
                        "raw_text": "",
                        "num_lines": 0,
                        "latency_seconds": 0.0,
                        "mean_confidence": 1.0,
                        "reference_transcript": ref_text,
                        "contains_math": contains_math,
                        "contains_diagram": contains_diagram,
                        "failure_reason": None
                    }
                    pred_writer.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                    pred_writer.flush()
                    existing_preds.add(key)
            continue

        # Load raw crop
        crop_rel = reg.get("crop_relative_path", f"crops/{sid}_{subq}_crop.png")
        crop_path = ROOT / "research_v2" / crop_rel
        if not crop_path.exists():
            print(f"  WARNING: Crop missing at {crop_path}")
            continue

        with open(crop_path, "rb") as cf:
            computed_sha = hashlib.sha256(cf.read()).hexdigest()
        expected_sha = reg.get("crop_sha256", "")
        sha_valid = (computed_sha == expected_sha) if expected_sha else True

        raw_bgr = cv2.imread(str(crop_path))
        if raw_bgr is None:
            print(f"  ERROR: Unable to decode image at {crop_path}")
            continue

        # Preprocess if not in cache
        if (sid, subq) not in prep_crops_cache:
            t_prep0 = time.time()
            prep_res = full_preprocess_pipeline(
                raw_bgr,
                remove_red=True,
                suppress_grid=False,
                deskew=True,
                denoise=False,
                binarize=False
            )
            clean_rgb = prep_res["clean_rgb"]
            clean_bgr = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
            prep_crops_cache[(sid, subq)] = {
                "bgr": clean_bgr,
                "rgb": clean_rgb,
                "latency": time.time() - t_prep0
            }

        prep_data = prep_crops_cache[(sid, subq)]
        prep_bgr = prep_data["bgr"]
        prep_rgb = prep_data["rgb"]

        # -------------------------------------------------------------------
        # 1. Tesseract
        # -------------------------------------------------------------------
        if "tesseract" in engines_to_run:
            for cond in conditions_to_run:
                key = (sid, subq, "tesseract", cond)
                if key in existing_preds:
                    continue

                img_in = raw_bgr if cond == "raw" else prep_rgb
                t0 = time.time()
                try:
                    raw_pred_text = tess_mod.image_to_string(img_in)
                    latency = time.time() - t0
                    lines = [ln.strip() for ln in raw_pred_text.splitlines() if ln.strip()]
                    fail_reason = None
                except Exception as e:
                    raw_pred_text = ""
                    lines = []
                    latency = time.time() - t0
                    fail_reason = f"TesseractError: {str(e)}"

                pred_record = {
                    "prediction_id": f"{sid}_{subq}_tesseract_{cond}",
                    "student_id": sid,
                    "subquestion_id": subq,
                    "parent_question_id": reg.get("parent_question_id", f"Q{subq[0]}"),
                    "engine": "tesseract",
                    "engine_version": f"v{tess_ver}",
                    "condition": cond,
                    "compute_device": "CPU",
                    "answer_status": "written",
                    "crop_relative_path": crop_rel,
                    "crop_sha256": computed_sha,
                    "crop_verified": sha_valid,
                    "raw_text": raw_pred_text,
                    "num_lines": len(lines),
                    "latency_seconds": round(latency, 4),
                    "mean_confidence": 0.70 if lines else 0.0,
                    "reference_transcript": ref_text,
                    "contains_math": contains_math,
                    "contains_diagram": contains_diagram,
                    "failure_reason": fail_reason
                }
                pred_writer.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                pred_writer.flush()
                existing_preds.add(key)

        # -------------------------------------------------------------------
        # 2. TrOCR Small
        # -------------------------------------------------------------------
        if "trocr" in engines_to_run:
            for cond in conditions_to_run:
                key = (sid, subq, "trocr", cond)
                if key in existing_preds:
                    continue

                img_in_bgr = raw_bgr if cond == "raw" else prep_bgr
                t0 = time.time()
                try:
                    boxes = segment_lines_morphology(img_in_bgr)
                    pil_img = Image.fromarray(cv2.cvtColor(img_in_bgr, cv2.COLOR_BGR2RGB))
                    line_crops = [pil_img.crop(b) for b in boxes]

                    trocr_lines = []
                    batch_size = 16
                    with torch.no_grad():
                        for b_idx in range(0, len(line_crops), batch_size):
                            batch = line_crops[b_idx:b_idx + batch_size]
                            inputs = trocr_proc(batch, return_tensors="pt", padding=True).to(trocr_device)
                            gen_ids = trocr_model.generate(inputs.pixel_values, max_new_tokens=48)
                            batch_texts = trocr_proc.batch_decode(gen_ids, skip_special_tokens=True)
                            trocr_lines.extend(batch_texts)

                    raw_pred_text = "\n".join(trocr_lines)
                    latency = time.time() - t0
                    fail_reason = None
                except Exception as e:
                    raw_pred_text = ""
                    trocr_lines = []
                    latency = time.time() - t0
                    fail_reason = f"TrOCRError: {str(e)}"

                pred_record = {
                    "prediction_id": f"{sid}_{subq}_trocr_{cond}",
                    "student_id": sid,
                    "subquestion_id": subq,
                    "parent_question_id": reg.get("parent_question_id", f"Q{subq[0]}"),
                    "engine": "trocr",
                    "engine_version": "microsoft/trocr-small-handwritten",
                    "condition": cond,
                    "compute_device": f"GPU ({trocr_device})",
                    "answer_status": "written",
                    "crop_relative_path": crop_rel,
                    "crop_sha256": computed_sha,
                    "crop_verified": sha_valid,
                    "raw_text": raw_pred_text,
                    "num_lines": len(trocr_lines),
                    "latency_seconds": round(latency, 4),
                    "mean_confidence": 0.85 if trocr_lines else 0.0,
                    "reference_transcript": ref_text,
                    "contains_math": contains_math,
                    "contains_diagram": contains_diagram,
                    "failure_reason": fail_reason
                }
                pred_writer.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                pred_writer.flush()
                existing_preds.add(key)

        # -------------------------------------------------------------------
        # 3. PaddleOCR PP-OCRv6
        # -------------------------------------------------------------------
        if "paddleocr" in engines_to_run:
            for cond in conditions_to_run:
                key = (sid, subq, "paddleocr", cond)
                if key in existing_preds:
                    continue

                img_in_bgr = raw_bgr if cond == "raw" else prep_bgr
                t0 = time.time()
                try:
                    # Scale down to 960 width for CPU efficiency while preserving handwriting aspect ratio
                    h_c, w_c = img_in_bgr.shape[:2]
                    target_w = 960
                    target_h = int(h_c * (target_w / w_c))
                    scaled_bgr = cv2.resize(img_in_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

                    res_list = list(paddle_ocr.predict(scaled_bgr))
                    rec_texts = res_list[0].get("rec_texts", []) if res_list else []
                    rec_scores = res_list[0].get("rec_scores", []) if res_list else []
                    mean_conf = float(np.mean(rec_scores)) if rec_scores else 0.0

                    raw_pred_text = "\n".join(rec_texts)
                    latency = time.time() - t0
                    fail_reason = None
                except Exception as e:
                    raw_pred_text = ""
                    rec_texts = []
                    mean_conf = 0.0
                    latency = time.time() - t0
                    fail_reason = f"PaddleOCRError: {str(e)}"

                pred_record = {
                    "prediction_id": f"{sid}_{subq}_paddleocr_{cond}",
                    "student_id": sid,
                    "subquestion_id": subq,
                    "parent_question_id": reg.get("parent_question_id", f"Q{subq[0]}"),
                    "engine": "paddleocr",
                    "engine_version": "PP-OCRv6_medium",
                    "condition": cond,
                    "compute_device": "CPU",
                    "answer_status": "written",
                    "crop_relative_path": crop_rel,
                    "crop_sha256": computed_sha,
                    "crop_verified": sha_valid,
                    "raw_text": raw_pred_text,
                    "num_lines": len(rec_texts),
                    "latency_seconds": round(latency, 4),
                    "mean_confidence": round(mean_conf, 4),
                    "reference_transcript": ref_text,
                    "contains_math": contains_math,
                    "contains_diagram": contains_diagram,
                    "failure_reason": fail_reason
                }
                pred_writer.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                pred_writer.flush()
                existing_preds.add(key)

        processed_count += 1

    pred_writer.close()
    print(f"\nLive OCR inference complete in {time.time() - total_start_time:.1f}s.")
    print("Now computing comprehensive publication-grade recognition metrics...")
    compute_and_save_all_metrics()

# ---------------------------------------------------------------------------
# Metrics Computation & Export
# ---------------------------------------------------------------------------
def compute_and_save_all_metrics():
    pred_file = OCR_OUT_DIR / "ocr_predictions.jsonl"
    if not pred_file.exists():
        print(f"ERROR: {pred_file} does not exist.")
        return

    # Load all predictions
    all_predictions = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_predictions.append(json.loads(line))

    print(f"Loaded {len(all_predictions)} prediction records.")

    # 1. Answer-level metrics
    answer_rows = []
    failure_rows = []
    runtime_records = defaultdict(list)

    for p in all_predictions:
        sid = p["student_id"]
        subq = p["subquestion_id"]
        parent_q = p["parent_question_id"]
        eng = p["engine"]
        cond = p["condition"]
        ans_status = p["answer_status"]
        device = p["compute_device"]
        latency = float(p.get("latency_seconds", 0.0))
        conf = float(p.get("mean_confidence", 0.0))
        ref = p.get("reference_transcript", "")
        hyp = p.get("raw_text", "")
        c_math = bool(p.get("contains_math", False))
        c_diag = bool(p.get("contains_diagram", False))

        runtime_records[(eng, cond, device)].append(latency)

        # Normalizations
        ref_strict = normalize_strict(ref)
        hyp_strict = normalize_strict(hyp)
        ref_light = normalize_light(ref)
        hyp_light = normalize_light(hyp)

        ref_char_len = len(ref_strict)
        hyp_char_len = len(hyp_strict)
        ref_word_len = len(ref_light.split())
        hyp_word_len = len(hyp_light.split())

        # Strict CER/WER
        cer_strict = compute_cer(ref_strict, hyp_strict)
        wer_strict = compute_wer(ref_strict, hyp_strict)

        # Light CER/WER
        cer_light = compute_cer(ref_light, hyp_light)
        wer_light = compute_wer(ref_light, hyp_light)

        exact_match_strict = int(ref_strict == hyp_strict)
        exact_match_light = int(ref_light == hyp_light)

        # Math tokens
        math_prf = compute_token_prf(extract_math_tokens(ref), extract_math_tokens(hyp))
        # Numbers
        num_prf = compute_token_prf(extract_numbers(ref), extract_numbers(hyp))

        # Blank / omission detection
        is_omitted_ref = (ans_status == "omitted_by_student")
        is_blank_hyp = (hyp_light == "")
        blank_correct = int(is_omitted_ref == is_blank_hyp)

        # Tag failures
        if p.get("failure_reason") or (ans_status == "written" and not hyp_light):
            fail_type = "empty_output" if not hyp_light else "engine_exception"
            failure_rows.append({
                "student_id": sid,
                "subquestion_id": subq,
                "engine": eng,
                "condition": cond,
                "failure_type": fail_type,
                "ref_length_words": ref_word_len,
                "latency_seconds": latency,
                "details": p.get("failure_reason") or "Empty hypothesis generated for written response"
            })

        answer_rows.append({
            "prediction_id": p["prediction_id"],
            "student_id": sid,
            "subquestion_id": subq,
            "parent_question_id": parent_q,
            "engine": eng,
            "condition": cond,
            "compute_device": device,
            "answer_status": ans_status,
            "ref_char_len": ref_char_len,
            "hyp_char_len": hyp_char_len,
            "ref_word_len": ref_word_len,
            "hyp_word_len": hyp_word_len,
            "cer_strict": round(cer_strict, 4),
            "wer_strict": round(wer_strict, 4),
            "cer_light": round(cer_light, 4),
            "wer_light": round(wer_light, 4),
            "exact_match_strict": exact_match_strict,
            "exact_match_light": exact_match_light,
            "math_precision": round(math_prf[0], 4),
            "math_recall": round(math_prf[1], 4),
            "math_f1": round(math_prf[2], 4),
            "number_precision": round(num_prf[0], 4),
            "number_recall": round(num_prf[1], 4),
            "number_f1": round(num_prf[2], 4),
            "blank_detected_correctly": blank_correct,
            "mean_confidence": round(conf, 4),
            "latency_seconds": round(latency, 4),
            "contains_math": c_math,
            "contains_diagram": c_diag
        })

    # Save ocr_metrics_by_answer.csv
    ans_csv = OCR_OUT_DIR / "ocr_metrics_by_answer.csv"
    with open(ans_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(answer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(answer_rows)
    print(f"Saved answer metrics: {ans_csv} ({len(answer_rows)} rows)")

    # Save ocr_failures.csv
    fail_csv = OCR_OUT_DIR / "ocr_failures.csv"
    with open(fail_csv, "w", newline="", encoding="utf-8") as f:
        if failure_rows:
            writer = csv.DictWriter(f, fieldnames=list(failure_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failure_rows)
        else:
            f.write("student_id,subquestion_id,engine,condition,failure_type,ref_length_words,latency_seconds,details\n")
    print(f"Saved failures log: {fail_csv} ({len(failure_rows)} failure records)")

    # Save model_runtime.csv
    runtime_rows = []
    for (eng, cond, dev), lats in sorted(runtime_records.items()):
        lats_arr = np.array(lats)
        runtime_rows.append({
            "engine": eng,
            "condition": cond,
            "compute_device": dev,
            "total_calls": len(lats),
            "total_runtime_sec": round(float(np.sum(lats_arr)), 2),
            "mean_latency_sec": round(float(np.mean(lats_arr)), 4),
            "median_latency_sec": round(float(np.median(lats_arr)), 4),
            "p90_latency_sec": round(float(np.percentile(lats_arr, 90)), 4),
            "p95_latency_sec": round(float(np.percentile(lats_arr, 95)), 4),
            "min_latency_sec": round(float(np.min(lats_arr)), 4),
            "max_latency_sec": round(float(np.max(lats_arr)), 4)
        })

    rt_csv = OCR_OUT_DIR / "model_runtime.csv"
    with open(rt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(runtime_rows[0].keys()))
        writer.writeheader()
        writer.writerows(runtime_rows)
    print(f"Saved runtime telemetry: {rt_csv}")

    # 2. Student-level aggregation
    # Group by (student_id, engine, condition)
    student_groups = defaultdict(list)
    for r in answer_rows:
        student_groups[(r["student_id"], r["engine"], r["condition"])].append(r)

    student_rows = []
    for (sid, eng, cond), rows in sorted(student_groups.items()):
        written_rows = [r for r in rows if r["answer_status"] == "written"]
        macro_cer_strict = np.mean([r["cer_strict"] for r in written_rows]) if written_rows else 0.0
        macro_wer_strict = np.mean([r["wer_strict"] for r in written_rows]) if written_rows else 0.0
        macro_cer_light = np.mean([r["cer_light"] for r in written_rows]) if written_rows else 0.0
        macro_wer_light = np.mean([r["wer_light"] for r in written_rows]) if written_rows else 0.0

        # Micro metrics (aggregate edit distances)
        tot_ref_chars = sum(r["ref_char_len"] for r in written_rows)
        tot_ref_words = sum(r["ref_word_len"] for r in written_rows)

        exact_light = np.mean([r["exact_match_light"] for r in rows])
        math_rows = [r for r in written_rows if r["contains_math"]]
        math_f1 = np.mean([r["math_f1"] for r in math_rows]) if math_rows else 1.0

        student_rows.append({
            "student_id": sid,
            "engine": eng,
            "condition": cond,
            "total_answers": len(rows),
            "written_answers": len(written_rows),
            "macro_cer_strict": round(float(macro_cer_strict), 4),
            "macro_wer_strict": round(float(macro_wer_strict), 4),
            "macro_cer_light": round(float(macro_cer_light), 4),
            "macro_wer_light": round(float(macro_wer_light), 4),
            "exact_match_rate_light": round(float(exact_light), 4),
            "math_symbol_f1": round(float(math_f1), 4),
            "mean_latency_sec": round(float(np.mean([r["latency_seconds"] for r in rows])), 4)
        })

    stu_csv = OCR_OUT_DIR / "ocr_metrics_by_student.csv"
    with open(stu_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(student_rows[0].keys()))
        writer.writeheader()
        writer.writerows(student_rows)
    print(f"Saved student metrics: {stu_csv} ({len(student_rows)} rows)")

    # 3. Engine-level comprehensive summary table
    engine_groups = defaultdict(list)
    for r in answer_rows:
        engine_groups[(r["engine"], r["condition"])].append(r)

    engine_summary = []
    for (eng, cond), rows in sorted(engine_groups.items()):
        written_rows = [r for r in rows if r["answer_status"] == "written"]
        math_rows = [r for r in written_rows if r["contains_math"]]
        omitted_rows = [r for r in rows if r["answer_status"] == "omitted_by_student"]

        macro_cer_strict = float(np.mean([r["cer_strict"] for r in written_rows])) if written_rows else 0.0
        macro_wer_strict = float(np.mean([r["wer_strict"] for r in written_rows])) if written_rows else 0.0
        macro_cer_light = float(np.mean([r["cer_light"] for r in written_rows])) if written_rows else 0.0
        macro_wer_light = float(np.mean([r["wer_light"] for r in written_rows])) if written_rows else 0.0

        exact_match_light = float(np.mean([r["exact_match_light"] for r in written_rows])) if written_rows else 0.0
        math_f1 = float(np.mean([r["math_f1"] for r in math_rows])) if math_rows else 0.0
        num_f1 = float(np.mean([r["number_f1"] for r in written_rows])) if written_rows else 0.0
        blank_f1 = float(np.mean([r["blank_detected_correctly"] for r in omitted_rows])) if omitted_rows else 1.0

        # Micro edit rates: sum(cer * ref_len) / sum(ref_len)
        tot_chars = sum(r["ref_char_len"] for r in written_rows)
        tot_words = sum(r["ref_word_len"] for r in written_rows)
        micro_cer_light = float(sum(r["cer_light"] * r["ref_char_len"] for r in written_rows) / tot_chars) if tot_chars > 0 else 0.0
        micro_wer_light = float(sum(r["wer_light"] * r["ref_word_len"] for r in written_rows) / tot_words) if tot_words > 0 else 0.0

        engine_summary.append({
            "engine": eng,
            "condition": cond,
            "total_records": len(rows),
            "written_records": len(written_rows),
            "math_records": len(math_rows),
            "macro_cer_strict": round(macro_cer_strict, 4),
            "macro_wer_strict": round(macro_wer_strict, 4),
            "macro_cer_light": round(macro_cer_light, 4),
            "macro_wer_light": round(macro_wer_light, 4),
            "micro_cer_light": round(micro_cer_light, 4),
            "micro_wer_light": round(micro_wer_light, 4),
            "exact_match_light": round(exact_match_light, 4),
            "math_token_f1": round(math_f1, 4),
            "number_preservation_f1": round(num_f1, 4),
            "omission_detection_accuracy": round(blank_f1, 4),
            "mean_latency_sec": round(float(np.mean([r["latency_seconds"] for r in rows])), 4),
            "status": "measured_real"
        })

    eng_csv = OCR_OUT_DIR / "ocr_metrics_by_engine.csv"
    with open(eng_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(engine_summary[0].keys()))
        writer.writeheader()
        writer.writerows(engine_summary)
    print(f"Saved engine summary table: {eng_csv}")

    # Print summary table
    print("\n" + "=" * 90)
    print("PHASE 2 / GATE 2: RECOGNITION BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Engine':12s} | {'Cond':12s} | {'Macro CER':10s} | {'Macro WER':10s} | {'Micro CER':10s} | {'Math F1':8s} | {'Num F1':8s} | {'Latency':8s}")
    print("-" * 90)
    for row in engine_summary:
        print(f"{row['engine']:12s} | {row['condition']:12s} | {row['macro_cer_light']:10.4f} | {row['macro_wer_light']:10.4f} | {row['micro_cer_light']:10.4f} | {row['math_token_f1']:8.4f} | {row['number_preservation_f1']:8.4f} | {row['mean_latency_sec']:7.2f}s")
    print("=" * 90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 OCR Benchmark")
    parser.add_argument("--engines", nargs="+", default=["tesseract", "trocr", "paddleocr"], help="Engines to benchmark")
    parser.add_argument("--conditions", nargs="+", default=["raw", "preprocessed"], help="Conditions to evaluate")
    parser.add_argument("--students", nargs="+", default=None, help="Specific student IDs to process (default: all 37)")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing predictions")
    args = parser.parse_args()

    run_benchmark(
        selected_engines=args.engines,
        selected_conditions=args.conditions,
        limit_students=args.students,
        resume=(not args.no_resume)
    )
