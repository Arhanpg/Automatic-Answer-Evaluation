"""
backend/evaluation.py -- Comprehensive Evaluation Suite for OCR and Automated Grading.

Implements all standardized academic metrics across Milestones M2, M3, M4, M7, M8, M9 & M10:
- OCR: CER, WER, Character Accuracy, Word Accuracy, Math Symbol Accuracy, Exact Expression Rate.
- Grading: MAE, RMSE, Normalized MAE, Exact Match Accuracy, Pearson r, Spearman rho, and Quadratic Weighted Kappa (QWK).
"""
import math
import numpy as np
from typing import List, Dict, Any, Union, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
import rapidfuzz

MATH_SYMBOLS = set("+=-\\/*^()[]{}|_0123456789<>.%eE")

def compute_ocr_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Computes CER, WER, Character Accuracy, Word Accuracy, Math Symbol Accuracy, and Exact Expression Rate."""
    if not predictions or not references:
        return {
            "cer": 0.0, "wer": 0.0, "char_accuracy": 1.0, "word_accuracy": 1.0,
            "math_symbol_accuracy": 1.0, "exact_expression_rate": 1.0
        }

    total_chars = 0
    total_char_edits = 0
    total_words = 0
    total_word_edits = 0
    total_math_symbols = 0
    correct_math_symbols = 0
    exact_matches = 0

    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip()
        ref_clean = ref.strip()

        # Character Error Rate (CER) via normalized edit distance
        char_dist = rapidfuzz.distance.Levenshtein.distance(pred_clean, ref_clean)
        denom_chars = max(len(ref_clean), len(pred_clean), 1)
        total_chars += denom_chars
        total_char_edits += char_dist

        # Word Error Rate (WER) via normalized edit distance
        pred_words = pred_clean.split()
        ref_words = ref_clean.split()
        word_dist = rapidfuzz.distance.Levenshtein.distance(pred_words, ref_words)
        denom_words = max(len(ref_words), len(pred_words), 1)
        total_words += denom_words
        total_word_edits += word_dist

        # Exact Expression Rate
        if pred_clean == ref_clean:
            exact_matches += 1

        # Math Symbol Accuracy
        ref_math = [c for c in ref_clean if c in MATH_SYMBOLS]
        pred_math = [c for c in pred_clean if c in MATH_SYMBOLS]
        denom_math = max(len(ref_math), len(pred_math), 1)
        total_math_symbols += denom_math
        math_dist = rapidfuzz.distance.Levenshtein.distance(pred_math, ref_math)
        correct_math_symbols += max(0, denom_math - math_dist)

    cer = min(1.0, float(total_char_edits / max(1, total_chars)))
    wer = min(1.0, float(total_word_edits / max(1, total_words)))
    char_acc = max(0.0, 1.0 - cer)
    word_acc = max(0.0, 1.0 - wer)
    math_acc = float(correct_math_symbols / max(1, total_math_symbols)) if total_math_symbols > 0 else 1.0
    exact_rate = float(exact_matches / len(predictions))

    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "char_accuracy": round(char_acc, 4),
        "word_accuracy": round(word_acc, 4),
        "math_symbol_accuracy": round(math_acc, 4),
        "exact_expression_rate": round(exact_rate, 4),
    }

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: float = 0.0, max_rating: float = 20.0) -> float:
    """Computes Quadratic Weighted Kappa (QWK) on scores discretized to half-marks."""
    # Discretize to integer bins of 0.5 step
    step = 0.5
    y_true_bins = np.round((np.clip(y_true, min_rating, max_rating) - min_rating) / step).astype(int)
    y_pred_bins = np.round((np.clip(y_pred, min_rating, max_rating) - min_rating) / step).astype(int)

    n_bins = int(round((max_rating - min_rating) / step)) + 1
    labels = list(range(n_bins))

    try:
        qwk = cohen_kappa_score(y_true_bins, y_pred_bins, labels=labels, weights="quadratic")
        if math.isnan(qwk):
            return 1.0 if np.all(y_true_bins == y_pred_bins) else 0.0
        return float(qwk)
    except Exception:
        return 0.0

def compute_grading_metrics(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray], max_marks_array: Optional[Union[List[float], np.ndarray]] = None) -> Dict[str, float]:
    """Computes comprehensive grading metrics: MAE, RMSE, Normalized MAE, Exact Match Accuracy, Pearson r, Spearman rho, and QWK."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if len(yt) == 0:
        return {
            "n": 0, "mae": 0.0, "rmse": 0.0, "normalized_mae": 0.0,
            "exact_match_accuracy": 0.0, "pearson_r": 0.0, "spearman_rho": 0.0, "qwk": 0.0
        }

    errors = np.abs(yt - yp)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    # Normalized MAE
    if max_marks_array is not None:
        max_arr = np.asarray(max_marks_array, dtype=float)
        max_arr = np.where(max_arr <= 0, 1.0, max_arr)
        norm_mae = float(np.mean(errors / max_arr))
    else:
        norm_mae = float(mae / 20.0)

    # Exact Match Accuracy (rounded to nearest 0.5 step)
    yp_rounded = np.round(yp * 2.0) / 2.0
    exact_match = float(np.mean(yp_rounded == yt))
    within_0_5 = float(np.mean(errors <= 0.5))
    within_1_0 = float(np.mean(errors <= 1.0))

    # Pearson & Spearman correlation
    if len(yt) > 1 and np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
        pr, _ = pearsonr(yt, yp)
        sr, _ = spearmanr(yt, yp)
        pearson_val = float(pr) if not math.isnan(pr) else 0.0
        spearman_val = float(sr) if not math.isnan(sr) else 0.0
    else:
        pearson_val = 1.0 if np.allclose(yt, yp) else 0.0
        spearman_val = 1.0 if np.allclose(yt, yp) else 0.0

    # Quadratic Weighted Kappa
    max_score = float(np.max(max_marks_array)) if max_marks_array is not None else 20.0
    qwk = quadratic_weighted_kappa(yt, yp, min_rating=0.0, max_rating=max_score)

    return {
        "n": int(len(yt)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "normalized_mae": round(norm_mae, 4),
        "exact_match_accuracy": round(exact_match, 4),
        "within_0_5_mark_accuracy": round(within_0_5, 4),
        "within_1_mark_accuracy": round(within_1_0, 4),
        "pearson_r": round(pearson_val, 4),
        "spearman_rho": round(spearman_val, 4),
        "qwk": round(qwk, 4),
    }
