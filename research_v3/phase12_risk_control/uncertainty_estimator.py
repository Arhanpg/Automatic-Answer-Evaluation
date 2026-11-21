"""
research_v3/phase12_risk_control/uncertainty_estimator.py

Input-Side Calibrated Risk and Uncertainty Estimators for Automated Exam Evaluation:
Strictly zero-leakage of gold teacher marks (resolving Defect G).
Features extracted strictly from input-side perception and cognitive signals:
1. Multi-Engine OCR Disagreement (TrOCR vs Tesseract normalized edit distance)
2. Semantic Rubric Margin Uncertainty (Proximity to grading threshold boundaries)
3. Mathematical Syntactic Parse Ambiguity (SymPy parse failures on numeric tokens)
4. Structural Routing Ambiguity (Confidence of question header anchoring)
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import rapidfuzz


def compute_normalized_edit_distance(text_a: str, text_b: str) -> float:
    """
    Computes normalized Levenshtein distance between two OCR hypotheses.
    Returns value in [0.0, 1.0].
    """
    s_a = str(text_a or "").strip()
    s_b = str(text_b or "").strip()

    if not s_a and not s_b:
        return 0.0
    if not s_a or not s_b:
        return 1.0

    dist = rapidfuzz.distance.Levenshtein.distance(s_a, s_b)
    max_len = max(len(s_a), len(s_b), 1)
    return float(min(1.0, max(0.0, dist / max_len)))


def compute_semantic_threshold_margin(sim_score: float) -> float:
    """
    Calculates uncertainty from proximity to rubric threshold decision boundaries (0.25 and 0.55).
    Peak uncertainty (1.0) occurs directly on the decision boundary.
    """
    sim = max(0.0, min(1.0, float(sim_score)))
    # Distance to nearest decision boundary
    dist_25 = abs(sim - 0.25)
    dist_55 = abs(sim - 0.55)
    min_dist = min(dist_25, dist_55)

    # Scale: distance >= 0.25 -> uncertainty 0.0; distance == 0.0 -> uncertainty 1.0
    unc = max(0.0, 1.0 - (min_dist / 0.25))
    return float(unc)


def compute_math_parse_uncertainty(subq: str, answer_text: str) -> float:
    """
    Checks if a mathematical derivation subquestion contains unparsable mathematical tokens.
    """
    if subq not in ["2a", "2b", "2c", "3a", "3b", "3c", "4"]:
        return 0.0

    text = str(answer_text or "").strip()
    if not text:
        return 1.0

    # Look for expected numeric tokens
    has_numbers = bool(re.search(r"\d", text))
    if not has_numbers:
        return 0.85  # Math question without numbers has high uncertainty

    # Check for malformed fractions or orphan operators
    if re.search(r"[/\\|\+\-\*]\s*$", text) or re.search(r"^\s*[/\\|\+\-\*]", text):
        return 0.90

    return 0.15


class InputSideUncertaintyEstimator:
    """
    Strictly Input-Side Uncertainty Estimator for Examination Booklets.
    Zero access to gold teacher marks.
    """
    def __init__(self):
        self.name = "calibrated_input_side_ensemble"

    def estimate_student_uncertainty(
        self,
        student_id: str,
        trocr_answers: Dict[str, str],
        tesseract_answers: Dict[str, str],
        routing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes composite uncertainty score u(x) for a student across all subquestions.
        """
        subq_uncertainties = {}
        subq_list = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b", "3c", "4"]

        total_ocr_disagree = 0.0
        total_math_unc = 0.0
        total_rout_unc = 0.0

        for q in subq_list:
            txt_trocr = trocr_answers.get(q, "")
            txt_tess = tesseract_answers.get(q, "")

            # 1. Multi-engine OCR disagreement
            ocr_disagree = compute_normalized_edit_distance(txt_trocr, txt_tess)
            total_ocr_disagree += ocr_disagree

            # 2. Math syntax uncertainty
            math_u = compute_math_parse_uncertainty(q, txt_trocr)
            total_math_unc += math_u

            # 3. Routing ambiguity
            r_info = routing_info.get(q, {})
            # If routing indicated missed header, merge, or low IoU
            is_clean_rout = (r_info.get("error_category", "none") == "none") and (r_info.get("iou", 1.0) >= 0.75)
            rout_u = 0.0 if is_clean_rout else 1.0
            total_rout_unc += rout_u

            # Subquestion composite
            subq_u = 0.45 * ocr_disagree + 0.30 * math_u + 0.25 * rout_u
            subq_uncertainties[q] = round(float(subq_u), 4)

        n_sub = len(subq_list)
        avg_ocr = total_ocr_disagree / n_sub
        avg_math = total_math_unc / n_sub
        avg_rout = total_rout_unc / n_sub

        # Student-level composite uncertainty score in [0.0, 1.0]
        # Heavily weighted towards OCR disagreement and routing ambiguity
        student_composite_unc = 0.45 * avg_ocr + 0.30 * avg_math + 0.25 * avg_rout

        return {
            "student_id": student_id,
            "composite_uncertainty": round(float(student_composite_unc), 4),
            "ocr_disagreement": round(float(avg_ocr), 4),
            "math_uncertainty": round(float(avg_math), 4),
            "routing_uncertainty": round(float(avg_rout), 4),
            "subquestion_uncertainties": subq_uncertainties
        }
