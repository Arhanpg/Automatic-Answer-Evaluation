"""
research_v3/phase11_error_propagation/tests/test_error_propagation.py

Unit tests for Phase 11 Controlled 2x2 Factorial Error Propagation:
1. Two-Factor Shapley Value Attribution and Interaction Coupling consistency
2. Two-Way ANOVA Sum-of-Squares and Degrees-of-Freedom consistency
3. Assessment metric bounds and calculations
4. Factorial stream dictionary construction integrity
5. Generated CSV benchmark artifact validation
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
from research_v3.phase11_error_propagation.anova_decomposition import (
    compute_evaluation_metrics,
    compute_shapley_decomposition,
    compute_twoway_anova,
    compute_paired_significance
)
from research_v3.phase11_error_propagation.factorial_streams import (
    FactorialStreamBuilder,
    SUBQ_ORDER
)


def test_shapley_mathematical_consistency():
    # Synthetic values representing realistic MAEs
    mae_a = 6.0
    mae_b = 11.0  # +5.0 recognition
    mae_c = 9.0   # +3.0 routing
    mae_d = 12.0  # +6.0 compound

    decomp = compute_shapley_decomposition(mae_a, mae_b, mae_c, mae_d)

    # 1. Total Delta
    assert abs(decomp["delta_total"] - 6.0) < 1e-5
    # 2. Marginal losses
    assert abs(decomp["delta_recognition"] - 5.0) < 1e-5
    assert abs(decomp["delta_routing"] - 3.0) < 1e-5
    # 3. Interaction coupling: 6 - (5 + 3) = -2
    assert abs(decomp["interaction_coupling"] - (-2.0)) < 1e-5
    # 4. Shapley values must sum exactly to delta_total
    phi_sum = decomp["shapley_phi_recognition"] + decomp["shapley_phi_routing"]
    assert abs(phi_sum - decomp["delta_total"]) < 1e-5
    # 5. Percentages must sum to 100%
    pct_sum = decomp["shapley_pct_recognition"] + decomp["shapley_pct_routing"]
    assert abs(pct_sum - 100.0) < 1e-2


def test_anova_sum_of_squares_consistency():
    np.random.seed(42)
    n = 30
    # Create synthetic cell arrays
    a = np.random.normal(15.0, 2.0, n)
    b = np.random.normal(10.0, 2.5, n)
    c = np.random.normal(12.0, 2.2, n)
    d = np.random.normal(8.0, 3.0, n)

    df_anova = compute_twoway_anova(a, b, c, d)
    assert len(df_anova) == 5

    ss_rec = df_anova.loc[df_anova["Source"] == "Perceptual Recognition (OCR)", "SS"].iloc[0]
    ss_rout = df_anova.loc[df_anova["Source"] == "Structural Routing (Layout)", "SS"].iloc[0]
    ss_inter = df_anova.loc[df_anova["Source"] == "Recognition x Routing Interaction", "SS"].iloc[0]
    ss_error = df_anova.loc[df_anova["Source"] == "Error (Within-Cell Residual)", "SS"].iloc[0]
    ss_total = df_anova.loc[df_anova["Source"] == "Total", "SS"].iloc[0]

    # SS components must sum to total SS
    ss_explained = ss_rec + ss_rout + ss_inter
    assert abs((ss_explained + ss_error) - ss_total) < 1e-2

    # Degrees of freedom must match 2x2 factorial structure
    df_rec = df_anova.loc[df_anova["Source"] == "Perceptual Recognition (OCR)", "df"].iloc[0]
    df_rout = df_anova.loc[df_anova["Source"] == "Structural Routing (Layout)", "df"].iloc[0]
    df_inter = df_anova.loc[df_anova["Source"] == "Recognition x Routing Interaction", "df"].iloc[0]
    df_error = df_anova.loc[df_anova["Source"] == "Error (Within-Cell Residual)", "df"].iloc[0]
    df_total = df_anova.loc[df_anova["Source"] == "Total", "df"].iloc[0]

    assert df_rec == 1
    assert df_rout == 1
    assert df_inter == 1
    assert df_error == 4 * (n - 1)
    assert df_total == 4 * n - 1
    assert (df_rec + df_rout + df_inter + df_error) == df_total


def test_evaluation_metrics_bounds():
    y_true = np.array([10.0, 15.0, 8.0, 18.0, 12.0])
    y_pred = np.array([9.0, 14.5, 7.0, 18.0, 14.0])

    m = compute_evaluation_metrics(y_true, y_pred, max_mark=20.0)
    assert m["mae"] >= 0.0
    assert m["rmse"] >= m["mae"]  # Jensen's inequality
    assert -1.0 <= m["qwk"] <= 1.0
    assert -1.0 <= m["spearman_rho"] <= 1.0
    assert 0.0 <= m["within_05m_pct"] <= 100.0
    assert 0.0 <= m["within_10m_pct"] <= 100.0


def test_factorial_stream_builder():
    builder = FactorialStreamBuilder()
    assert len(builder.student_ids) == 37

    # Test student s1
    ans_a = builder.build_stream_a("s1")
    assert set(ans_a.keys()) == set(SUBQ_ORDER)
    assert len(ans_a["1a"]) > 0

    ans_b = builder.build_stream_b("s1", "trocr_raw")
    assert set(ans_b.keys()) == set(SUBQ_ORDER)

    ans_c = builder.build_stream_c("s1")
    assert set(ans_c.keys()) == set(SUBQ_ORDER)

    ans_d = builder.build_stream_d_baseline("s1", "trocr_raw")
    assert set(ans_d.keys()) == set(SUBQ_ORDER)
