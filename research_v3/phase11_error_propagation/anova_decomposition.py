"""
research_v3/phase11_error_propagation/anova_decomposition.py

Statistical Analysis and Variance Decomposition for the 2x2 Factorial Experiment:
1. Two-Factor Shapley Value Attribution (Recognition vs Routing)
2. Non-linear Interaction Coupling (I)
3. Formal Two-Way Factorial ANOVA (SS, df, MS, F-statistic, p-value, partial eta-squared)
4. Evaluation Metrics (MAE, RMSE, MBE, QWK, Spearman rho, Pearson r, Tolerances)
5. Paired Statistical Significance (Wilcoxon signed-rank tests, Paired t-tests, Cohen's d)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_mark: float = 20.0
) -> Dict[str, float]:
    """
    Computes comprehensive assessment evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    if n == 0:
        return {}

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mbe = float(np.mean(errors))

    exact_pct = float(np.mean(abs_errors == 0.0) * 100.0)
    within_05_pct = float(np.mean(abs_errors <= 0.5) * 100.0)
    within_10_pct = float(np.mean(abs_errors <= 1.0) * 100.0)

    # Asymmetric penalties
    severe_under = float(np.mean((y_true - y_pred) >= 3.0) * 100.0)
    severe_over = float(np.mean((y_pred - y_true) >= 3.0) * 100.0)

    # Quadratic Weighted Kappa
    # Discretize marks into integer half-mark bins
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)), 0.0)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)), max_mark)
    bins = np.linspace(min_val, max_val, num=int(max_mark * 2) + 1)
    y_t_disc = np.digitize(y_true, bins)
    y_p_disc = np.digitize(y_pred, bins)

    try:
        qwk = float(cohen_kappa_score(y_t_disc, y_p_disc, weights="quadratic"))
    except Exception:
        qwk = 0.0

    # Correlations
    try:
        spearman_rho, _ = stats.spearmanr(y_true, y_pred)
        spearman_rho = float(spearman_rho) if not np.isnan(spearman_rho) else 0.0
    except Exception:
        spearman_rho = 0.0

    try:
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        pearson_r = float(pearson_r) if not np.isnan(pearson_r) else 0.0
    except Exception:
        pearson_r = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mbe": round(mbe, 4),
        "exact_match_pct": round(exact_pct, 2),
        "within_05m_pct": round(within_05_pct, 2),
        "within_10m_pct": round(within_10_pct, 2),
        "severe_under_marking_pct": round(severe_under, 2),
        "severe_over_marking_pct": round(severe_over, 2),
        "qwk": round(qwk, 4),
        "spearman_rho": round(spearman_rho, 4),
        "pearson_r": round(pearson_r, 4)
    }


def compute_shapley_decomposition(
    mae_a: float,
    mae_b: float,
    mae_c: float,
    mae_d: float
) -> Dict[str, float]:
    """
    Computes Two-Factor Shapley Value Attribution and Interaction Coupling.
    Stream A: (0, 0) - Human, Oracle
    Stream B: (1, 0) - OCR, Oracle
    Stream C: (0, 1) - Human, Baseline Routing
    Stream D: (1, 1) - OCR, Baseline Routing
    """
    delta_total = max(1e-6, mae_d - mae_a)
    delta_rec = mae_b - mae_a
    delta_rout = mae_c - mae_a

    # Interaction coupling
    interaction = delta_total - (delta_rec + delta_rout)

    # Shapley values
    phi_rec = 0.5 * delta_rec + 0.5 * (delta_total - delta_rout)
    phi_rout = 0.5 * delta_rout + 0.5 * (delta_total - delta_rec)

    pct_rec = (phi_rec / delta_total) * 100.0
    pct_rout = (phi_rout / delta_total) * 100.0

    return {
        "delta_total": round(delta_total, 4),
        "delta_recognition": round(delta_rec, 4),
        "delta_routing": round(delta_rout, 4),
        "interaction_coupling": round(interaction, 4),
        "shapley_phi_recognition": round(phi_rec, 4),
        "shapley_phi_routing": round(phi_rout, 4),
        "shapley_pct_recognition": round(pct_rec, 2),
        "shapley_pct_routing": round(pct_rout, 2)
    }


def compute_twoway_anova(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    scores_c: np.ndarray,
    scores_d: np.ndarray
) -> pd.DataFrame:
    """
    Computes formal Two-Way Factorial ANOVA on student grading marks.
    Factor 1: Recognition (Human = 0, OCR = 1)
    Factor 2: Routing (Oracle = 0, Baseline = 1)
    Each cell has N observations (N=37).
    """
    n = len(scores_a)
    assert len(scores_b) == n and len(scores_c) == n and len(scores_d) == n

    # Cell data
    y = np.concatenate([scores_a, scores_b, scores_c, scores_d])
    n_total = len(y)  # 4 * n
    grand_mean = np.mean(y)

    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    mean_c = np.mean(scores_c)
    mean_d = np.mean(scores_d)

    # Marginal means
    mean_rec_0 = 0.5 * (mean_a + mean_c)  # Human
    mean_rec_1 = 0.5 * (mean_b + mean_d)  # OCR
    mean_rout_0 = 0.5 * (mean_a + mean_b)  # Oracle
    mean_rout_1 = 0.5 * (mean_c + mean_d)  # Baseline

    # Sum of Squares
    ss_total = np.sum((y - grand_mean) ** 2)

    # SS Recognition: 2 * n * [(mean_rec_0 - grand_mean)^2 + (mean_rec_1 - grand_mean)^2]
    ss_rec = 2 * n * ((mean_rec_0 - grand_mean) ** 2 + (mean_rec_1 - grand_mean) ** 2)

    # SS Routing: 2 * n * [(mean_rout_0 - grand_mean)^2 + (mean_rout_1 - grand_mean)^2]
    ss_rout = 2 * n * ((mean_rout_0 - grand_mean) ** 2 + (mean_rout_1 - grand_mean) ** 2)

    # SS Subcells: n * [(mean_a - grand_mean)^2 + ...]
    ss_subcells = n * (
        (mean_a - grand_mean) ** 2 +
        (mean_b - grand_mean) ** 2 +
        (mean_c - grand_mean) ** 2 +
        (mean_d - grand_mean) ** 2
    )

    # SS Interaction: SS_subcells - SS_rec - SS_rout
    ss_inter = max(0.0, ss_subcells - ss_rec - ss_rout)

    # SS Error (Within cells): sum over each cell of (y_i - cell_mean)^2
    ss_error = (
        np.sum((scores_a - mean_a) ** 2) +
        np.sum((scores_b - mean_b) ** 2) +
        np.sum((scores_c - mean_c) ** 2) +
        np.sum((scores_d - mean_d) ** 2)
    )

    # Degrees of freedom
    df_rec = 1
    df_rout = 1
    df_inter = 1
    df_error = 4 * (n - 1)
    df_total = n_total - 1

    # Mean Squares
    ms_rec = ss_rec / df_rec
    ms_rout = ss_rout / df_rout
    ms_inter = ss_inter / df_inter
    ms_error = ss_error / df_error

    # F-statistics and p-values
    f_rec = ms_rec / ms_error
    f_rout = ms_rout / ms_error
    f_inter = ms_inter / ms_error

    p_rec = float(stats.f.sf(f_rec, df_rec, df_error))
    p_rout = float(stats.f.sf(f_rout, df_rout, df_error))
    p_inter = float(stats.f.sf(f_inter, df_inter, df_error))

    # Partial eta-squared
    eta_rec = ss_rec / (ss_rec + ss_error)
    eta_rout = ss_rout / (ss_rout + ss_error)
    eta_inter = ss_inter / (ss_inter + ss_error)

    rows = [
        {"Source": "Perceptual Recognition (OCR)", "SS": round(ss_rec, 4), "df": df_rec, "MS": round(ms_rec, 4), "F": round(f_rec, 4), "p_value": p_rec, "partial_eta_sq": round(eta_rec, 4)},
        {"Source": "Structural Routing (Layout)", "SS": round(ss_rout, 4), "df": df_rout, "MS": round(ms_rout, 4), "F": round(f_rout, 4), "p_value": p_rout, "partial_eta_sq": round(eta_rout, 4)},
        {"Source": "Recognition x Routing Interaction", "SS": round(ss_inter, 4), "df": df_inter, "MS": round(ms_inter, 4), "F": round(f_inter, 4), "p_value": p_inter, "partial_eta_sq": round(eta_inter, 4)},
        {"Source": "Error (Within-Cell Residual)", "SS": round(ss_error, 4), "df": df_error, "MS": round(ms_error, 4), "F": np.nan, "p_value": np.nan, "partial_eta_sq": np.nan},
        {"Source": "Total", "SS": round(ss_total, 4), "df": df_total, "MS": round(ss_total / df_total, 4), "F": np.nan, "p_value": np.nan, "partial_eta_sq": np.nan}
    ]
    return pd.DataFrame(rows)


def compute_paired_significance(
    base_scores: np.ndarray,
    comp_scores: np.ndarray
) -> Dict[str, float]:
    """Computes paired t-test, Wilcoxon signed-rank test, and Cohen's d effect size."""
    diff = comp_scores - base_scores
    diff_std = np.std(diff, ddof=1)
    cohen_d = float(np.mean(diff) / diff_std) if diff_std > 0 else 0.0

    try:
        t_stat, p_ttest = stats.ttest_rel(comp_scores, base_scores)
    except Exception:
        t_stat, p_ttest = 0.0, 1.0

    try:
        wilc_stat, p_wilcoxon = stats.wilcoxon(comp_scores, base_scores)
    except Exception:
        wilc_stat, p_wilcoxon = 0.0, 1.0

    return {
        "mean_difference": round(float(np.mean(diff)), 4),
        "std_difference": round(float(diff_std), 4),
        "cohen_d": round(cohen_d, 4),
        "paired_t_stat": round(float(t_stat), 4),
        "paired_t_p": float(p_ttest),
        "wilcoxon_stat": round(float(wilc_stat), 4),
        "wilcoxon_p": float(p_wilcoxon)
    }
