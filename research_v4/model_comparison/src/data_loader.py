"""
research_v4/model_comparison/src/data_loader.py

Leakage-Safe Institutional Dataset Loader:
Loads 37 student booklets, 148 parent-question responses, schemas, teacher marks,
and authentic Phase 9 multi-engine OCR predictions into structured records.
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
SCHEMA_PATH = REPO_ROOT / "schemas" / "v1" / "exam_schema.json"
GT_TRANSCRIPTS_PATH = DATA_DIR / "ground_truth" / "verified_transcriptions.json"
TEACHER_MARKS_DIR = DATA_DIR / "teacher_marks"
PHASE9_OCR_PATH = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking" / "results" / "predictions.jsonl"

SUBQUESTIONS_BY_PARENT = {
    "1": ["1a", "1b", "1c", "1d"],
    "2": ["2a", "2b", "2c"],
    "3": ["3a", "3b", "3c"],
    "4": ["4"]
}


def load_exam_schema() -> Dict[str, Any]:
    """Loads canonical exam schema with rubrics and question criteria."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Exam schema missing at {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_teacher_marks() -> Dict[str, Dict[str, float]]:
    """Loads authentic single-marker teacher marks for all students."""
    marks_by_student = {}
    for p in glob.glob(str(TEACHER_MARKS_DIR / "s*.json")):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            sid = data.get("roll_number", Path(p).stem)
            marks_by_student[sid] = {str(k): float(v) for k, v in data.get("marks", {}).items()}
    return marks_by_student


def load_phase9_ocr_hypotheses() -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads authentic per-subquestion OCR predictions from Phase 9.
    Structure: hypotheses[engine_condition][student_id][subquestion_id] = hypothesis_text
    """
    hypotheses = {
        "trocr_actual": {},
        "tesseract_actual": {},
        "paddle_actual": {}
    }

    if not PHASE9_OCR_PATH.exists():
        return hypotheses

    with open(PHASE9_OCR_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            eng = rec.get("engine")
            cond = rec.get("condition")
            sid = rec.get("student_id")
            sq = rec.get("subquestion_id")
            hyp = rec.get("hypothesis", "")

            target_key = None
            if eng == "trocr" and cond == "raw":
                target_key = "trocr_actual"
            elif eng == "tesseract" and cond == "preprocessed":
                target_key = "tesseract_actual"
            elif eng == "paddleocr" and cond == "raw":
                target_key = "paddle_actual"

            if target_key:
                if sid not in hypotheses[target_key]:
                    hypotheses[target_key][sid] = {}
                hypotheses[target_key][sid][sq] = hyp

    return hypotheses


def load_evaluation_dataset() -> List[Dict[str, Any]]:
    """
    Builds the complete list of 148 parent question evaluations (37 students x 4 questions).
    Every record contains clean reference text, authentic OCR text, teacher mark, and schema metadata.
    """
    schema = load_exam_schema()
    teacher_marks = load_all_teacher_marks()
    ocr_hypotheses = load_phase9_ocr_hypotheses()

    if not GT_TRANSCRIPTS_PATH.exists():
        raise FileNotFoundError(f"Verified transcriptions missing at {GT_TRANSCRIPTS_PATH}")
    with open(GT_TRANSCRIPTS_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # Index schema by question id
    schema_q = {str(q["id"]): q for q in schema.get("questions", [])}

    records = []
    student_ids = [f"s{i}" for i in range(1, 38)]

    for sid in student_ids:
        s_gt = gt_data.get(sid, {}).get("answers", {})
        s_tm = teacher_marks.get(sid, {})

        for qid in ["1", "2", "3", "4"]:
            q_meta = schema_q.get(qid, {})
            max_m = float(q_meta.get("max_marks", q_meta.get("marks", 0.0)))
            t_mark = float(s_tm.get(qid, 0.0))

            subqs = SUBQUESTIONS_BY_PARENT[qid]

            # 1. Corrected Reference Text
            sq_corrected = {sq: s_gt.get(sq, "") for sq in subqs}
            text_corrected = " ".join([v for v in sq_corrected.values() if v.strip()]).strip()

            # 2. Authentic TrOCR Text
            sq_trocr = {sq: ocr_hypotheses["trocr_actual"].get(sid, {}).get(sq, "") for sq in subqs}
            text_trocr = " ".join([v for v in sq_trocr.values() if v.strip()]).strip()
            # If completely empty from OCR drop, default to empty string
            if not text_trocr:
                text_trocr = ""

            # 3. Authentic Tesseract Text
            sq_tess = {sq: ocr_hypotheses["tesseract_actual"].get(sid, {}).get(sq, "") for sq in subqs}
            text_tess = " ".join([v for v in sq_tess.values() if v.strip()]).strip()

            # Schema criteria & model answer
            criteria = q_meta.get("rubric", q_meta.get("criteria", []))
            model_ans = str(q_meta.get("model_answer", q_meta.get("reference_solution", "")))

            record = {
                "response_id": f"{sid}_Q{qid}",
                "student_id": sid,
                "question_id": qid,
                "max_marks": max_m,
                "teacher_mark": t_mark,
                "is_math_question": 1.0 if qid in ["2", "4"] else 0.0,
                # Text input representations
                "corrected_transcript": text_corrected,
                "subquestions_corrected": sq_corrected,
                "trocr_actual": text_trocr,
                "subquestions_trocr": sq_trocr,
                "tesseract_actual": text_tess,
                "subquestions_tesseract": sq_tess,
                # Rubric & reference
                "model_answer": model_ans,
                "rubric_criteria": criteria,
                "prompt_text": q_meta.get("prompt", q_meta.get("text", f"Question {qid}"))
            }
            records.append(record)

    return records
