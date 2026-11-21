"""
research_v3/phase12_risk_control/selective_review.py

Selective Human Review & Workload-Risk Optimization:
Simulates human-in-the-loop triaging across realistic institutional review budgets:
B in [0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%, 60%, 75%, 100%].
Compares:
1. Proposed Input-Side Calibrated Ensemble (Ours - Zero Label Leakage)
2. OCR Multi-Engine Disagreement Only
3. Routing Ambiguity Only
4. Random Selection Baseline (Expected Value over 100 Random Trials)
5. Oracle Error Ranking (Theoretical Maximum Cheating Bound)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def evaluate_residual_metrics(
    y_true: np.ndarray,
    y_final: np.ndarray,
    max_mark: float = 20.0
) -> Dict[str, float]:
    """Computes residual MAE, RMSE, QWK, and agreement percentages."""
    abs_errors = np.abs(y_final - y_true)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_final - y_true) ** 2)))
    within_05 = float(np.mean(abs_errors <= 0.5) * 100.0)
    within_10 = float(np.mean(abs_errors <= 1.0) * 100.0)

    # QWK
    bins = np.linspace(0.0, max_mark, num=int(max_mark * 2) + 1)
    y_t_disc = np.digitize(y_true, bins)
    y_f_disc = np.digitize(y_final, bins)
    try:
        qwk = float(cohen_kappa_score(y_t_disc, y_f_disc, weights="quadratic"))
    except Exception:
        qwk = 0.0

    return {
        "residual_MAE": round(mae, 4),
        "residual_RMSE": round(rmse, 4),
        "residual_QWK": round(qwk, 4),
        "within_05m_pct": round(within_05, 2),
        "within_10m_pct": round(within_10, 2)
    }


def simulate_selective_review(
    student_records: List[Dict[str, Any]],
    ranking_key: str,
    reverse: bool = True,
    budget_fractions: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Ranks student records by ranking_key and applies selective human review.
    """
    if budget_fractions is None:
        budget_fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]

    n_total = len(student_records)
    # Sort records
    sorted_recs = sorted(student_records, key=lambda r: r[ranking_key], reverse=reverse)

    curve_rows = []
    for budget in budget_fractions:
        k_reviewed = int(round(budget * n_total))
        workload_saved = round((1.0 - (k_reviewed / n_total)) * 100.0, 1)

        y_true_list = []
        y_final_list = []

        for idx, rec in enumerate(sorted_recs):
            y_t = float(rec["teacher_total"])
            y_p = float(rec["predicted_total"])
            y_true_list.append(y_t)

            if idx < k_reviewed:
                # Referred to human reviewer: perfect adjudication
                y_final_list.append(y_t)
            else:
                # Processed autonomously
                y_final_list.append(y_p)

        y_true_arr = np.array(y_true_list)
        y_final_arr = np.array(y_final_list)

        met = evaluate_residual_metrics(y_true_arr, y_final_arr)

        curve_rows.append({
            "review_budget_fraction": budget,
            "review_budget_pct": round(budget * 100.0, 1),
            "scripts_reviewed": k_reviewed,
            "scripts_autonomous": n_total - k_reviewed,
            "workload_saved_pct": workload_saved,
            "residual_MAE": met["residual_MAE"],
            "residual_RMSE": met["residual_RMSE"],
            "residual_QWK": met["residual_QWK"],
            "within_05m_pct": met["within_05m_pct"],
            "within_10m_pct": met["within_10m_pct"]
        })

    return curve_rows


def simulate_random_selective_review(
    student_records: List[Dict[str, Any]],
    n_trials: int = 100,
    budget_fractions: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """Simulates expected random selection over n_trials."""
    if budget_fractions is None:
        budget_fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]

    n_total = len(student_records)
    curve_rows = []

    for budget in budget_fractions:
        k_reviewed = int(round(budget * n_total))
        workload_saved = round((1.0 - (k_reviewed / n_total)) * 100.0, 1)

        trial_maes = []
        trial_rmses = []
        trial_qwks = []
        trial_w05s = []
        trial_w10s = []

        for seed in range(n_trials):
            rng = np.random.RandomState(seed)
            shuffled_indices = rng.permutation(n_total)

            y_true_list = []
            y_final_list = []

            for rank_idx, orig_idx in enumerate(shuffled_indices):
                rec = student_records[orig_idx]
                y_t = float(rec["teacher_total"])
                y_p = float(rec["predicted_total"])
                y_true_list.append(y_t)

                if rank_idx < k_reviewed:
                    y_final_list.append(y_t)
                else:
                    y_final_list.append(y_p)

            met = evaluate_residual_metrics(np.array(y_true_list), np.array(y_final_list))
            trial_maes.append(met["residual_MAE"])
            trial_rmses.append(met["residual_RMSE"])
            trial_qwks.append(met["residual_QWK"])
            trial_w05s.append(met["within_05m_pct"])
            trial_w10s.append(met["within_10m_pct"])

        curve_rows.append({
            "review_budget_fraction": budget,
            "review_budget_pct": round(budget * 100.0, 1),
            "scripts_reviewed": k_reviewed,
            "scripts_autonomous": n_total - k_reviewed,
            "workload_saved_pct": workload_saved,
            "residual_MAE": round(float(np.mean(trial_maes)), 4),
            "residual_RMSE": round(float(np.mean(trial_rmses)), 4),
            "residual_QWK": round(float(np.mean(trial_qwks)), 4),
            "within_05m_pct": round(float(np.mean(trial_w05s)), 2),
            "within_10m_pct": round(float(np.mean(trial_w10s)), 2)
        })

    return curve_rows
