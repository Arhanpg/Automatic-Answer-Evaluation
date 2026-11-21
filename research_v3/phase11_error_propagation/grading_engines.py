"""
research_v3/phase11_error_propagation/grading_engines.py

Implementations of the 4 Grading Engines evaluated across the 2x2 Factorial Streams:
1. UnifiedHybridGrader (Criterion-level semantic S-BERT + SymPy symbolic math + rule constraints)
2. DenseSemanticGrader (Pure Sentence-BERT cosine similarity against model rubrics)
3. SymbolicMathGrader (Pure SymPy symbolic verification for mathematical derivations)
4. LexicalRuleGrader (Pure lexical TF-IDF, token overlap, and regex value matching)
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
from backend import criterion_aggregator, semantic_scorer, math_verifier

PARENT_MAX_MARKS = {"Q1": 8.0, "Q2": 6.0, "Q3": 3.0, "Q4": 3.0, "total": 20.0}
SUBQ_TO_PARENT = {
    "1a": "Q1", "1b": "Q1", "1c": "Q1", "1d": "Q1",
    "2a": "Q2", "2b": "Q2", "2c": "Q2",
    "3a": "Q3", "3b": "Q3", "3c": "Q3",
    "4": "Q4"
}


class UnifiedHybridGrader:
    """
    Proposed State-of-the-Art Hybrid Grader.
    Integrates:
    - Dense semantic sentence embeddings (all-MiniLM-L6-v2)
    - SymPy exact algebraic and numerical equivalence
    - Domain rule overrides and directional contradiction detection
    """
    def __init__(self):
        self.name = "unified_hybrid_grader"

    def grade_student(self, student_id: str, answers: Dict[str, str]) -> Dict[str, float]:
        res = criterion_aggregator.grade_student_answers(student_id, answers)
        q_scores = res["question_scores"]
        return {
            "Q1": float(q_scores["Q1"]),
            "Q2": float(q_scores["Q2"]),
            "Q3": float(q_scores["Q3"]),
            "Q4": float(q_scores["Q4"]),
            "total": float(q_scores["total"])
        }


class DenseSemanticGrader:
    """
    Pure Dense Semantic Grader Baseline.
    Evaluates student answers solely by S-BERT cosine similarity against official model answers.
    """
    def __init__(self):
        self.name = "dense_semantic_sbert"
        self.scorer = semantic_scorer._GLOBAL_DENSE_SCORER
        self.schema = criterion_aggregator.load_exam_schema()

    def grade_student(self, student_id: str, answers: Dict[str, str]) -> Dict[str, float]:
        scores = {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0, "total": 0.0}

        for q in self.schema["questions"]:
            qid = f"Q{q['id']}"
            q_awarded = 0.0
            for part in q["parts"]:
                pid = part["part_id"]
                ans_text = str(answers.get(pid, "")).strip()
                if not ans_text:
                    continue
                sim = self.scorer.compute_similarity(ans_text, part["model_answer"])
                if sim >= 0.55:
                    awarded = float(part["max_marks"])
                elif sim >= 0.25:
                    awarded = round(float(part["max_marks"]) * 0.5, 1)
                else:
                    awarded = 0.0
                q_awarded += awarded
            scores[qid] = min(PARENT_MAX_MARKS[qid], q_awarded)

        scores["total"] = scores["Q1"] + scores["Q2"] + scores["Q3"] + scores["Q4"]
        return scores


class SymbolicMathGrader:
    """
    Pure Symbolic Math Verifier Baseline.
    Uses SymPy to test exact mathematical and numerical correctness on Q2, Q3, Q4.
    Awards 0 on conceptual prose questions (Q1).
    """
    def __init__(self):
        self.name = "symbolic_sympy_verifier"

    def grade_student(self, student_id: str, answers: Dict[str, str]) -> Dict[str, float]:
        scores = {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0, "total": 0.0}
        math_results = math_verifier.verify_student_math(answers)
        scores["Q2"] = min(6.0, float(math_results["Q2"]["marks"]))
        scores["Q3"] = min(3.0, float(math_results["Q3"]["marks"]))
        scores["Q4"] = min(3.0, float(math_results["Q4"]["marks"]))
        scores["total"] = scores["Q2"] + scores["Q3"] + scores["Q4"]
        return scores


class LexicalRuleGrader:
    """
    Pure Lexical and Rule-Based Grader Baseline.
    Uses token Jaccard similarity for Q1 and regex value matching for Q2, Q3, Q4.
    """
    def __init__(self):
        self.name = "lexical_rule_grader"
        self.q1_model = {
            "1a": "n-gram Markov assumption fixed context window long-range dependencies",
            "1b": "permutation-invariant parallel self-attention positional encoding word order",
            "1c": "bidirectional encoder masked language model classification representations",
            "1d": "causal mask autoregressive upper-triangular future tokens information leakage"
        }

    def grade_student(self, student_id: str, answers: Dict[str, str]) -> Dict[str, float]:
        scores = {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0, "total": 0.0}

        # Q1 Lexical
        q1_tot = 0.0
        for pid in ["1a", "1b", "1c", "1d"]:
            ans = str(answers.get(pid, "")).strip()
            if not ans:
                continue
            sim = semantic_scorer.lexical_similarity(ans, self.q1_model[pid])
            part_score = round(sim["token_jaccard"] * 3.0, 1)
            part_score = min(2.0, max(0.0, part_score))
            q1_tot += part_score
        scores["Q1"] = min(8.0, q1_tot)

        # Q2 Regex
        ans_2a = str(answers.get("2a", ""))
        ans_2b = str(answers.get("2b", ""))
        ans_2c = str(answers.get("2c", ""))
        q2_tot = 0.0
        if "1" in ans_2a and "0" in ans_2a:
            q2_tot += 2.0
        if "0.73" in ans_2b or "0.27" in ans_2b or "exp" in ans_2b:
            q2_tot += 2.0
        if "1.46" in ans_2c or "0.81" in ans_2c or "v1" in ans_2c.lower():
            q2_tot += 2.0
        scores["Q2"] = min(6.0, q2_tot)

        # Q3 Regex
        ans_3a = str(answers.get("3a", ""))
        ans_3b = str(answers.get("3b", ""))
        ans_3c = str(answers.get("3c", ""))
        q3_tot = 0.0
        if "word a" in ans_3a.lower() or "0.4" in ans_3a:
            q3_tot += 1.0
        if "word a" in ans_3b.lower() and "word b" in ans_3b.lower():
            q3_tot += 1.0
        if "norm" in ans_3c.lower() or "sampling" in ans_3c.lower() or "nucleus" in ans_3c.lower():
            q3_tot += 1.0
        scores["Q3"] = min(3.0, q3_tot)

        # Q4 Regex
        ans_4 = str(answers.get("4", ""))
        q4_tot = 0.0
        if "2420" in ans_4 or "1050" in ans_4 or "12100" in ans_4 or "smoothing" in ans_4.lower():
            q4_tot += 3.0
        elif "3" in ans_4 or "laplace" in ans_4.lower() or "count" in ans_4.lower():
            q4_tot += 1.5
        scores["Q4"] = min(3.0, q4_tot)

        scores["total"] = scores["Q1"] + scores["Q2"] + scores["Q3"] + scores["Q4"]
        return scores
