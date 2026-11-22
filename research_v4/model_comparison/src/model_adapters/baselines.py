"""
research_v4/model_comparison/src/model_adapters/baselines.py

Non-Learning and Deterministic Rule Baseline Adapters:
1. MeanPredictor: predicts question-level mean mark from training fold.
2. MedianPredictor: predicts question-level median mark from training fold.
3. RubricCoverageBaseline: maps criterion coverage ratio to question max marks.
4. LexicalCalibrator: fits linear regression on training fold ROUGE-L / Jaccard similarity.
5. DeterministicRuleGrader: symbolic math verifier and rule-based criterion scoring.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import Ridge
from ..evaluation.metrics import quantize_half_mark


class MeanPredictor:
    """Predicts the mean training mark for each parent question."""

    def __init__(self):
        self.question_means: Dict[str, float] = {}

    def fit(self, records: List[Dict[str, Any]]) -> "MeanPredictor":
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for r in records:
            qid = str(r["question_id"])
            m = float(r["teacher_mark"])
            sums[qid] = sums.get(qid, 0.0) + m
            counts[qid] = counts.get(qid, 0) + 1

        self.question_means = {qid: sums[qid] / counts[qid] for qid in sums}
        return self

    def predict(self, records: List[Dict[str, Any]]) -> np.ndarray:
        preds = []
        for r in records:
            qid = str(r["question_id"])
            max_m = float(r["max_marks"])
            val = self.question_means.get(qid, max_m * 0.5)
            preds.append(val)
        return np.array(preds, dtype=np.float64)


class MedianPredictor:
    """Predicts the median training mark for each parent question."""

    def __init__(self):
        self.question_medians: Dict[str, float] = {}

    def fit(self, records: List[Dict[str, Any]]) -> "MedianPredictor":
        marks_by_q: Dict[str, List[float]] = {}
        for r in records:
            qid = str(r["question_id"])
            if qid not in marks_by_q:
                marks_by_q[qid] = []
            marks_by_q[qid].append(float(r["teacher_mark"]))

        self.question_medians = {qid: float(np.median(marks_by_q[qid])) for qid in marks_by_q}
        return self

    def predict(self, records: List[Dict[str, Any]]) -> np.ndarray:
        preds = []
        for r in records:
            qid = str(r["question_id"])
            max_m = float(r["max_marks"])
            val = self.question_medians.get(qid, max_m * 0.5)
            preds.append(val)
        return np.array(preds, dtype=np.float64)


class RubricCoverageBaseline:
    """Assigns marks proportional to semantic criterion coverage."""

    def __init__(self):
        pass

    def fit(self, records: List[Dict[str, Any]]) -> "RubricCoverageBaseline":
        return self

    def predict(self, X: np.ndarray, feature_names: List[str], records: List[Dict[str, Any]]) -> np.ndarray:
        preds = []
        cov_idx = feature_names.index("sbert_coverage_ratio") if "sbert_coverage_ratio" in feature_names else None

        for i, r in enumerate(records):
            max_m = float(r["max_marks"])
            if cov_idx is not None:
                cov = float(X[i, cov_idx])
            else:
                cov = 0.5
            preds.append(cov * max_m)
        return np.array(preds, dtype=np.float64)


class LexicalCalibrator:
    """Calibrates lexical similarities (ROUGE-L, Jaccard) to marks via linear model fit on train fold."""

    def __init__(self):
        self.model = Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> "LexicalCalibrator":
        # Extract lexical columns
        lex_cols = [i for i, name in enumerate(feature_names) if name.startswith("lex_")]
        if not lex_cols:
            lex_cols = [0]
        self.lex_indices = lex_cols
        self.model.fit(X[:, self.lex_indices], y)
        return self

    def predict(self, X: np.ndarray, records: List[Dict[str, Any]]) -> np.ndarray:
        raw_preds = self.model.predict(X[:, self.lex_indices])
        max_marks = np.array([float(r["max_marks"]) for r in records])
        return np.clip(raw_preds, 0.0, max_marks)


class DeterministicRuleGrader:
    """
    Combines SymPy mathematical verifiers with deterministic rubric keyword rules.
    """

    def __init__(self):
        pass

    def fit(self, records: List[Dict[str, Any]]) -> "DeterministicRuleGrader":
        return self

    def predict(self, X: np.ndarray, feature_names: List[str], records: List[Dict[str, Any]]) -> np.ndarray:
        sym_idx = feature_names.index("sym_raw_score") if "sym_raw_score" in feature_names else None
        sem_idx = feature_names.index("sbert_criteria_mean") if "sbert_criteria_mean" in feature_names else None

        preds = []
        for i, r in enumerate(records):
            qid = str(r["question_id"])
            max_m = float(r["max_marks"])
            if qid in ["2", "4"]:
                # Math question: prioritize symbolic verification
                val = float(X[i, sym_idx]) if sym_idx is not None else max_m * 0.5
            else:
                # Conceptual prose: scale semantic coverage
                cov = float(X[i, sem_idx]) if sem_idx is not None else 0.5
                val = cov * max_m
            preds.append(min(max_m, max(0.0, val)))
        return np.array(preds, dtype=np.float64)
