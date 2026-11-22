"""
research_v4/model_comparison/src/model_adapters/ensemble.py

Hybrid and Stacking Ensemble Adapters:
1. ValidationTunedStacking: Ridge meta-regressor trained on out-of-fold predictions.
2. ParetoOptimalEnsemble: Question-conditional routing to domain specialists
   (e.g., SymPy for math derivations, GBDT for conceptual prose).
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.linear_model import Ridge


class ValidationTunedStacking:
    """Combines diverse base model predictions using a validation-fitted meta-learner."""

    def __init__(self, alpha: float = 1.0):
        self.meta_learner = Ridge(alpha=alpha, positive=True, fit_intercept=True)
        self.fitted = False

    def fit(self, base_preds_train: np.ndarray, y_train: np.ndarray) -> "ValidationTunedStacking":
        """
        Fits meta-learner on matrix of base predictions [N_samples, N_models].
        """
        self.meta_learner.fit(base_preds_train, y_train)
        self.fitted = True
        return self

    def predict(self, base_preds_test: np.ndarray, max_marks: np.ndarray) -> np.ndarray:
        if not self.fitted:
            # Fallback to simple average
            avg = np.mean(base_preds_test, axis=1)
            return np.clip(avg, 0.0, max_marks)

        preds = self.meta_learner.predict(base_preds_test)
        return np.clip(preds, 0.0, max_marks)


class ParetoOptimalEnsemble:
    """
    Routes each question to its validated domain specialist:
    Q1 (Prose): Tree-based GBDT / LightGBM
    Q2 (Attention Math): Symbolic SymPy Verifier
    Q3 (Decoding Algorithms): Hybrid RAG / Random Forest
    Q4 (Statistical Smoothing): Symbolic Verifier + Ridge
    """

    def __init__(self):
        pass

    def predict(
        self,
        records: List[Dict[str, Any]],
        model_predictions_dict: Dict[str, np.ndarray],
        max_marks: np.ndarray
    ) -> np.ndarray:
        final_preds = []

        gbdt_preds = model_predictions_dict.get("lightgbm_regressor", model_predictions_dict.get("ridge_regression"))
        sympy_preds = model_predictions_dict.get("deterministic_rule_grader", gbdt_preds)
        rag_preds = model_predictions_dict.get("hybrid_bm25_dense_rag", gbdt_preds)
        ridge_preds = model_predictions_dict.get("ridge_regression", gbdt_preds)

        for i, r in enumerate(records):
            qid = str(r["question_id"])
            if qid == "1":
                # Conceptual prose: Tree-based GBDT
                p = gbdt_preds[i]
            elif qid == "2":
                # Attention matrix derivation: SymPy symbolic
                p = sympy_preds[i]
            elif qid == "3":
                # Decoding algorithm: Hybrid RAG
                p = rag_preds[i]
            else:  # 4
                # Statistical Laplace: Blend SymPy and Ridge
                p = 0.5 * sympy_preds[i] + 0.5 * ridge_preds[i]
            final_preds.append(p)

        return np.clip(np.array(final_preds, dtype=np.float64), 0.0, max_marks)
