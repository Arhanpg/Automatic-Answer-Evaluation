"""
research_v3/phase11_error_propagation/factorial_streams.py

Constructs the 5 Evaluation Streams for the 2x2 Factorial Experiment:
- Stream A: Human Transcript + Oracle Routing (Grader Upper Bound Control)
- Stream B (TrOCR): TrOCR-Small Raw OCR + Oracle Routing (Pure Recognition Error)
- Stream B (Tesseract): Tesseract Preprocessed OCR + Oracle Routing
- Stream C: Human Transcript + Baseline Regex Routing (Pure Routing Error)
- Stream D (Baseline): TrOCR-Small Raw OCR + Baseline Regex Routing (Compound Reality)
- Stream D (Hybrid): TrOCR-Small Raw OCR + Hybrid Layout Routing (Proposed Recovery)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ADJUDICATIONS_JSON = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "annotations_store" / "adjudications.json"
TRANSCRIPTS_CSV = REPO_ROOT / "research_v2" / "annotations" / "transcripts.csv"
OCR_PRED_JSONL = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking" / "results" / "predictions.jsonl"
ROUTING_CSV = REPO_ROOT / "research_v3" / "phase10_routing" / "results" / "routing_comparison.csv"

SUBQ_ORDER = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b", "3c", "4"]


def load_all_transcripts() -> pd.DataFrame:
    """Loads verified human ground truth transcripts from Phase 8 consensus."""
    if ADJUDICATIONS_JSON.exists():
        with open(ADJUDICATIONS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for k, v in data.items():
            status = v.get("final_adjudicated_status", "")
            is_omitted = "omitted" in status
            rows.append({
                "student_id": v["student_id"],
                "subquestion_id": v["subquestion_id"],
                "answer_status": "omitted_by_student" if is_omitted else "student_answer",
                "adjudicated_transcript": v.get("final_adjudicated_transcript", "")
            })
        return pd.DataFrame(rows)
    df = pd.read_csv(TRANSCRIPTS_CSV)
    df["adjudicated_transcript"] = df["adjudicated_transcript"].fillna("").astype(str)
    return df


def load_ocr_predictions() -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Loads OCR hypotheses grouped by (engine_condition) -> student_id -> subquestion_id -> text.
    """
    data = {}
    with open(OCR_PRED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = f"{rec['engine']}_{rec['condition']}"
            data.setdefault(key, {}).setdefault(rec["student_id"], {})[rec["subquestion_id"]] = rec.get("hypothesis", "")
    return data


def load_routing_records() -> pd.DataFrame:
    """Loads Phase 10 routing comparison records."""
    return pd.read_csv(ROUTING_CSV)


class FactorialStreamBuilder:
    """Builds question-answer dictionaries for every student across all factorial streams."""

    def __init__(self):
        self.trans_df = load_all_transcripts()
        self.ocr_data = load_ocr_predictions()
        self.routing_df = load_routing_records()
        self.student_ids = sorted(self.trans_df["student_id"].unique(), key=lambda s: int(s.replace("s", "")))

    def build_stream_a(self, student_id: str) -> Dict[str, str]:
        """Stream A: Human Ground Truth + Oracle Routing."""
        sub = self.trans_df[self.trans_df["student_id"] == student_id]
        answers = {q: "" for q in SUBQ_ORDER}
        for _, row in sub.iterrows():
            if row["answer_status"] != "omitted_by_student":
                answers[row["subquestion_id"]] = row["adjudicated_transcript"]
        return answers

    def build_stream_b(self, student_id: str, engine_key: str = "trocr_raw") -> Dict[str, str]:
        """Stream B: OCR Hypothesis + Oracle Routing (Pure Recognition Error)."""
        answers = {q: "" for q in SUBQ_ORDER}
        student_ocr = self.ocr_data.get(engine_key, {}).get(student_id, {})
        for q in SUBQ_ORDER:
            answers[q] = student_ocr.get(q, "")
        return answers

    def build_stream_c(self, student_id: str) -> Dict[str, str]:
        """Stream C: Human Ground Truth + Baseline Regex Routing (Pure Routing Error)."""
        answers = {q: "" for q in SUBQ_ORDER}
        gt_answers = self.build_stream_a(student_id)

        # Apply sheet_parser_baseline routing mapping
        sub_rout = self.routing_df[
            (self.routing_df["student_id"] == student_id) &
            (self.routing_df["system_id"] == "sheet_parser_baseline")
        ]

        for _, row in sub_rout.iterrows():
            gt_q = row["subquestion_id"]
            pred_q = str(row["pred_subquestion_id"]) if pd.notna(row["pred_subquestion_id"]) else None
            text = gt_answers.get(gt_q, "")

            if pred_q and pred_q in answers:
                # Merge into predicted slot
                if answers[pred_q]:
                    answers[pred_q] += "\n" + text
                else:
                    answers[pred_q] = text
        return answers

    def build_stream_d_baseline(self, student_id: str, engine_key: str = "trocr_raw") -> Dict[str, str]:
        """Stream D Baseline: OCR Hypothesis + Baseline Regex Routing (End-to-End Baseline)."""
        answers = {q: "" for q in SUBQ_ORDER}
        ocr_answers = self.build_stream_b(student_id, engine_key)

        sub_rout = self.routing_df[
            (self.routing_df["student_id"] == student_id) &
            (self.routing_df["system_id"] == "sheet_parser_baseline")
        ]

        for _, row in sub_rout.iterrows():
            gt_q = row["subquestion_id"]
            pred_q = str(row["pred_subquestion_id"]) if pd.notna(row["pred_subquestion_id"]) else None
            text = ocr_answers.get(gt_q, "")

            if pred_q and pred_q in answers:
                if answers[pred_q]:
                    answers[pred_q] += "\n" + text
                else:
                    answers[pred_q] = text
        return answers

    def build_stream_d_hybrid(self, student_id: str, engine_key: str = "trocr_raw") -> Dict[str, str]:
        """Stream D Hybrid: OCR Hypothesis + Hybrid Layout Routing (Proposed Recovery)."""
        answers = {q: "" for q in SUBQ_ORDER}
        ocr_answers = self.build_stream_b(student_id, engine_key)

        sub_rout = self.routing_df[
            (self.routing_df["student_id"] == student_id) &
            (self.routing_df["system_id"] == "hybrid_layout_continuation")
        ]

        for _, row in sub_rout.iterrows():
            gt_q = row["subquestion_id"]
            pred_q = str(row["pred_subquestion_id"]) if pd.notna(row["pred_subquestion_id"]) else None
            text = ocr_answers.get(gt_q, "")

            if pred_q and pred_q in answers:
                if answers[pred_q]:
                    answers[pred_q] += "\n" + text
                else:
                    answers[pred_q] = text
        return answers
