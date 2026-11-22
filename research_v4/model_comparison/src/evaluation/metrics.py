"""
research_v4/model_comparison/src/evaluation/metrics.py

Comprehensive Evaluation Metrics Suite:
1. Quadratic Weighted Kappa (QWK) on standardized half-mark increments.
2. Mean Absolute Error (MAE), RMSE, Normalized MAE (N-MAE).
3. Within +-0.5 mark, Within +-1.0 mark, and Exact Agreement percentages.
4. Pearson r and Spearman rho rank correlation.
"""

from typing import Dict, Any, List, Union
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def quantize_half_mark(scores: np.ndarray, max_marks: Union[float, np.ndarray]) -> np.ndarray:
    """Clips scores to [0, max_marks] and rounds to nearest half-mark (0.5)."""
    clipped = np.clip(scores, 0.0, max_marks)
    return np.round(clipped * 2.0) / 2.0


def compute_qwk_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_rating: float = 0.0,
    max_rating: float = 8.0
) -> float:
    """
    Computes Quadratic Weighted Kappa on discretized half-mark integer ratings.
    """
    # Convert half-marks (0.0, 0.5, 1.0, ...) to integer rating categories (0, 1, 2, ...)
    yt_cat = np.round(y_true * 2.0).astype(int)
    yp_cat = np.round(y_pred * 2.0).astype(int)

    min_cat = int(min_rating * 2.0)
    max_cat = int(max_rating * 2.0)

    # Force categories into [min_cat, max_cat]
    yt_cat = np.clip(yt_cat, min_cat, max_cat)
    yp_cat = np.clip(yp_cat, min_cat, max_cat)

    try:
        categories = list(range(min_cat, max_cat + 1))
        qwk = cohen_kappa_score(yt_cat, yp_cat, weights="quadratic", labels=categories)
        if np.isnan(qwk):
            return 0.0
        return float(qwk)
    except Exception:
        return 0.0


def compute_grading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_marks: np.ndarray
) -> Dict[str, float]:
    """
    Computes all standard ASAG grading metrics across a set of responses.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mm = np.asarray(max_marks, dtype=np.float64)

    # Quantize predictions to legal half-mark increments
    yp_quant = quantize_half_mark(yp, mm)

    abs_err = np.abs(yt - yp_quant)
    sq_err = np.square(yt - yp_quant)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(sq_err)))
    norm_mae = float(np.mean(abs_err / np.maximum(1.0, mm)))

    within_05 = float(np.mean(abs_err <= 0.50) * 100.0)
    within_10 = float(np.mean(abs_err <= 1.00) * 100.0)
    exact_acc = float(np.mean(abs_err == 0.0) * 100.0)

    # QWK
    qwk = compute_qwk_score(yt, yp_quant, min_rating=0.0, max_rating=float(np.max(mm)))

    # Correlations
    if np.std(yt) > 1e-6 and np.std(yp_quant) > 1e-6:
        pr, _ = stats.pearsonr(yt, yp_quant)
        sr, _ = stats.spearmanr(yt, yp_quant)
        pr = float(pr) if not np.isnan(pr) else 0.0
        sr = float(sr) if not np.isnan(sr) else 0.0
    else:
        pr, sr = 0.0, 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "normalized_mae": round(norm_mae, 4),
        "qwk": round(qwk, 4),
        "within_05_pct": round(within_05, 2),
        "within_10_pct": round(within_10, 2),
        "exact_match_pct": round(exact_acc, 2),
        "pearson_r": round(pr, 4),
        "spearman_rho": round(sr, 4)
    }
