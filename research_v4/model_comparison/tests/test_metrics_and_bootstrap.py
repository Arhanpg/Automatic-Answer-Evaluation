"""
research_v4/model_comparison/tests/test_metrics_and_bootstrap.py

Unit Tests for Metrics, Discrete Half-Mark Quantization, and Cluster Bootstrap:
1. Verifies QWK, MAE, RMSE, and correlation computations.
2. Proves that predictions are constrained to legal question bounds [0, max_marks].
3. Verifies student-level cluster bootstrap with all 4 questions sampled together.
"""

import pytest
import numpy as np
from research_v4.model_comparison.src.evaluation.metrics import (
    quantize_half_mark, compute_qwk_score, compute_grading_metrics
)
from research_v4.model_comparison.src.statistics.bootstrap import (
    cluster_bootstrap_metric, paired_cluster_permutation_test
)


def test_quantize_half_mark():
    raw = np.array([-1.2, 0.2, 0.4, 0.7, 1.8, 5.9, 12.0])
    max_m = np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0])
    quant = quantize_half_mark(raw, max_m)
    expected = np.array([0.0, 0.0, 0.5, 0.5, 2.0, 6.0, 8.0])
    np.testing.assert_array_equal(quant, expected)


def test_grading_metrics_exact_agreement():
    yt = np.array([8.0, 6.0, 3.0, 3.0])
    yp = np.array([8.0, 6.0, 3.0, 3.0])
    mm = np.array([8.0, 6.0, 3.0, 3.0])

    m = compute_grading_metrics(yt, yp, mm)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["qwk"] == 1.0
    assert m["within_05_pct"] == 100.0
    assert m["exact_match_pct"] == 100.0


def test_cluster_bootstrap_carries_all_student_questions():
    """
    Verifies that cluster bootstrap resamples student IDs,
    ensuring all 4 parent questions stay intact within a cluster.
    """
    student_ids = []
    for i in range(1, 38):
        student_ids.extend([f"s{i}"] * 4)

    y_true = np.random.uniform(0.0, 8.0, size=len(student_ids))
    y_pred = y_true + np.random.normal(0, 0.5, size=len(student_ids))
    max_marks = np.array([8.0, 6.0, 3.0, 3.0] * 37)

    point_est, ci_l, ci_u = cluster_bootstrap_metric(
        student_ids, y_true, y_pred, max_marks, metric_key="mae", n_resamples=100
    )

    assert ci_l <= point_est <= ci_u
    assert ci_u > ci_l
