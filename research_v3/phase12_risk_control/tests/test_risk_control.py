"""
research_v3/phase12_risk_control/tests/test_risk_control.py

Unit tests for Phase 12 Input-Side Calibrated Risk Control & Selective Human Review:
1. Input-side metric bounds and calculations (zero label leakage)
2. Semantic margin decision boundary uncertainty
3. Selective review monotonicity and endpoint constraints (0% -> MAE_base, 100% -> MAE=0.0)
4. Generated CSV benchmark artifact validation
"""

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np
import pandas as pd

from research_v3.phase12_risk_control.uncertainty_estimator import (
    compute_normalized_edit_distance,
    compute_semantic_threshold_margin,
    compute_math_parse_uncertainty,
    InputSideUncertaintyEstimator
)
from research_v3.phase12_risk_control.selective_review import (
    evaluate_residual_metrics,
    simulate_selective_review
)


def test_normalized_edit_distance_bounds():
    assert compute_normalized_edit_distance("hello world", "hello world") == 0.0
    assert compute_normalized_edit_distance("", "") == 0.0
    assert compute_normalized_edit_distance("abc", "") == 1.0
    assert compute_normalized_edit_distance("", "xyz") == 1.0

    d = compute_normalized_edit_distance("hello world", "hella warld")
    assert 0.0 < d < 1.0


def test_semantic_threshold_margin():
    # Peak uncertainty directly on decision boundaries (0.25 and 0.55)
    assert abs(compute_semantic_threshold_margin(0.25) - 1.0) < 1e-4
    assert abs(compute_semantic_threshold_margin(0.55) - 1.0) < 1e-4

    # Low uncertainty far from decision boundaries
    assert compute_semantic_threshold_margin(0.0) == 0.0
    assert compute_semantic_threshold_margin(1.0) == 0.0


def test_math_parse_uncertainty():
    # Non-math question returns 0.0
    assert compute_math_parse_uncertainty("1a", "n-gram models") == 0.0

    # Math question with numbers returns low uncertainty
    assert compute_math_parse_uncertainty("2b", "softmax is 0.731 and 0.269") == 0.15

    # Math question without numbers returns high uncertainty
    assert compute_math_parse_uncertainty("2b", "softmax is purely based on attention") == 0.85

    # Empty text returns 1.0
    assert compute_math_parse_uncertainty("4", "") == 1.0


def test_selective_review_monotonicity():
    # Create synthetic student cohort
    n = 20
    np.random.seed(42)
    records = []
    for i in range(n):
        y_t = float(np.random.randint(5, 18))
        err = float(np.random.uniform(1.0, 10.0))
        y_p = y_t + err
        u = err / 10.0 + np.random.uniform(0.0, 0.2)
        records.append({
            "student_id": f"s{i+1}",
            "teacher_total": y_t,
            "predicted_total": y_p,
            "uncertainty": u
        })

    curve = simulate_selective_review(records, ranking_key="uncertainty", reverse=True)
    maes = [r["residual_MAE"] for r in curve]

    # At 0% review, residual MAE is positive
    assert maes[0] > 0.0
    # At 100% review, residual MAE is exactly 0.0
    assert maes[-1] == 0.0

    # Monotonicity: each step of human review must reduce or maintain MAE
    for j in range(len(maes) - 1):
        assert maes[j] >= maes[j + 1]


def test_risk_control_artifacts_integrity():
    res_dir = REPO_ROOT / "research_v3" / "phase12_risk_control" / "results"

    curves_file = res_dir / "selective_review_curves.csv"
    corr_file = res_dir / "calibration_correlation.csv"
    unc_file = res_dir / "student_uncertainty_scores.csv"
    sub_file = res_dir / "subgroup_risk_profile.csv"

    assert curves_file.exists(), f"Missing {curves_file}"
    assert corr_file.exists(), f"Missing {corr_file}"
    assert unc_file.exists(), f"Missing {unc_file}"
    assert sub_file.exists(), f"Missing {sub_file}"

    df_unc = pd.read_csv(unc_file)
    assert len(df_unc) == 37, f"Expected 37 students, got {len(df_unc)}"
    assert (df_unc["composite_uncertainty"] >= 0.0).all()
    assert (df_unc["composite_uncertainty"] <= 1.0).all()

    df_curves = pd.read_csv(curves_file)
    assert len(df_curves) == 60  # 5 strategies x 12 budgets
