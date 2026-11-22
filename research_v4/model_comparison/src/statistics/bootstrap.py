"""
research_v4/model_comparison/src/statistics/bootstrap.py

Student-Clustered Statistical Inference Suite:
1. Clustered Bootstrap Confidence Intervals (B = 2,000 resamples at student level, carrying all 4 questions).
2. Clustered Paired Permutation Tests on student-level error deltas.
3. Holm-Bonferroni Multiplicity Correction.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from ..evaluation.metrics import compute_grading_metrics, quantize_half_mark


def get_student_clusters(student_ids: List[str]) -> Dict[str, List[int]]:
    """Groups dataset indices by student ID."""
    clusters = {}
    for idx, sid in enumerate(student_ids):
        if sid not in clusters:
            clusters[sid] = []
        clusters[sid].append(idx)
    return clusters


def cluster_bootstrap_metric(
    student_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_marks: np.ndarray,
    metric_key: str = "mae",
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes 95% confidence intervals using cluster bootstrap resampled at the student level.
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    clusters = get_student_clusters(student_ids)
    unique_students = list(clusters.keys())
    n_clusters = len(unique_students)

    base_metrics = compute_grading_metrics(y_true, y_pred, max_marks)
    point_est = float(base_metrics.get(metric_key, 0.0))

    boot_stats = []
    for _ in range(n_resamples):
        sampled_students = rng.choice(unique_students, size=n_clusters, replace=True)
        boot_indices = []
        for s in sampled_students:
            boot_indices.extend(clusters[s])

        b_yt = y_true[boot_indices]
        b_yp = y_pred[boot_indices]
        b_mm = max_marks[boot_indices]

        m = compute_grading_metrics(b_yt, b_yp, b_mm)
        boot_stats.append(m.get(metric_key, 0.0))

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))

    return point_est, round(ci_lower, 4), round(ci_upper, 4)


def paired_cluster_permutation_test(
    student_ids: List[str],
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    max_marks: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Two-tailed paired cluster permutation test comparing per-student MAE between Model A and Model B.
    """
    rng = np.random.RandomState(seed)
    clusters = get_student_clusters(student_ids)
    unique_students = list(clusters.keys())

    yp_a = quantize_half_mark(y_pred_a, max_marks)
    yp_b = quantize_half_mark(y_pred_b, max_marks)

    # Compute mean absolute error per student cluster
    student_mae_a = []
    student_mae_b = []
    for s in unique_students:
        idx = clusters[s]
        err_a = np.mean(np.abs(y_true[idx] - yp_a[idx]))
        err_b = np.mean(np.abs(y_true[idx] - yp_b[idx]))
        student_mae_a.append(err_a)
        student_mae_b.append(err_b)

    arr_a = np.array(student_mae_a)
    arr_b = np.array(student_mae_b)
    diff = arr_a - arr_b
    obs_diff = float(np.mean(diff))

    # Effect size (Cohen's d)
    std_diff = np.std(diff, ddof=1)
    cohen_d = float(obs_diff / std_diff) if std_diff > 1e-6 else 0.0

    # Permutation test
    count_extreme = 0
    for _ in range(n_permutations):
        # Random sign flip for each student cluster
        signs = rng.choice([-1.0, 1.0], size=len(unique_students))
        perm_diff = np.mean(diff * signs)
        if abs(perm_diff) >= abs(obs_diff):
            count_extreme += 1

    p_val = float(count_extreme / n_permutations)

    return {
        "mean_delta_mae": round(obs_diff, 4),
        "cohens_d": round(cohen_d, 4),
        "p_value": round(p_val, 5),
        "statistically_significant_05": p_val < 0.05
    }


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction for multiple comparisons."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(sorted_pairs):
        adj_p = p * (m - rank)
        adj_p = min(1.0, max(running_max, adj_p))
        running_max = adj_p
        adjusted[orig_idx] = round(adj_p, 5)

    return adjusted
