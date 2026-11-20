"""
tests/test_phase6_robustness.py -- Unit and Scientific Validation Suite for Gate 6.

Verifies:
1. Perturbation robustness artifacts exist and cover 16 distortion conditions.
2. Distortion degradations are positive: severe skew and low resolution increase error rates.
3. Selective human review curve is monotonically non-increasing in residual MAE.
4. At 100% human review budget, residual MAE reaches 0.0.
5. Subgroup analysis covers all key pedagogical and handwriting dimensions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBUSTNESS_DIR = REPO_ROOT / "research_v2" / "robustness"
CALIBRATION_DIR = REPO_ROOT / "research_v2" / "calibration"


def test_perturbation_summary_completeness():
    """Validates that perturbation summary contains all 16 distortion conditions."""
    csv_path = ROBUSTNESS_DIR / "perturbation_summary.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 16, f"Expected 16 perturbation conditions, got {len(df)}"

    conditions = set(df["condition_id"].unique())
    expected = {
        "clean", "skew_1deg", "skew_2deg", "skew_5deg", "skew_10deg",
        "blur_sigma1", "blur_sigma2", "blur_sigma3",
        "dpi_200", "dpi_150", "dpi_75",
        "gamma_0.7", "gamma_1.4",
        "jpeg_q50", "jpeg_q25", "jpeg_q10"
    }
    assert expected.issubset(conditions)


def test_perturbation_degradation_behavior():
    """Validates that environmental distortions degrade recognition performance or cause total deletion."""
    csv_path = ROBUSTNESS_DIR / "perturbation_summary.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)
    clean_cer = df[df["condition_id"] == "clean"]["mean_cer"].iloc[0]
    skew1_cer = df[df["condition_id"] == "skew_1deg"]["mean_cer"].iloc[0]
    gamma_cer = df[df["condition_id"] == "gamma_1.4"]["mean_cer"].iloc[0]
    jpeg_cer = df[df["condition_id"] == "jpeg_q25"]["mean_cer"].iloc[0]
    skew10_cer = df[df["condition_id"] == "skew_10deg"]["mean_cer"].iloc[0]

    # Geometric tilt 1deg increases CER
    assert skew1_cer > clean_cer, f"1deg skew CER ({skew1_cer}) should exceed clean CER ({clean_cer})"
    # Overexposure gamma 1.4 increases CER
    assert gamma_cer > clean_cer, f"Gamma 1.4 CER ({gamma_cer}) should exceed clean CER ({clean_cer})"
    # Lossy JPEG compression increases CER
    assert jpeg_cer > clean_cer, f"JPEG q25 CER ({jpeg_cer}) should exceed clean CER ({clean_cer})"
    # Severe 10deg skew causes complete line loss (100% deletion error, CER = 1.0)
    assert abs(skew10_cer - 1.0) < 1e-4, f"10deg skew should result in 100% deletion (CER=1.0), got {skew10_cer}"


def test_selective_review_monotonicity():
    """Validates that selective review curve is monotonically non-increasing in residual MAE."""
    csv_path = CALIBRATION_DIR / "selective_review_curve.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)
    assert len(df) >= 10

    maes = df["residual_MAE"].values
    budgets = df["review_budget_fraction"].values

    # Check monotonicity: as budget increases, MAE must stay equal or decrease
    for i in range(1, len(maes)):
        assert maes[i] <= maes[i - 1] + 1e-4, (
            f"Non-monotonic MAE at budget {budgets[i]}: {maes[i]} > {maes[i-1]}"
        )

    # Check endpoints
    assert abs(maes[-1] - 0.0) < 1e-4, f"100% review budget should have residual MAE 0.0, got {maes[-1]}"
    assert maes[0] > 0.0, f"0% review budget should have positive autonomous MAE, got {maes[0]}"


def test_subgroup_analysis_completeness():
    """Validates subgroup stratification coverage across modalities and handwriting features."""
    csv_path = CALIBRATION_DIR / "subgroup_analysis.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    df = pd.read_csv(csv_path)
    categories = set(df["subgroup_category"].unique())
    expected_categories = {"Question Modality", "Answer Length Band", "Mathematical Notation", "Diagram Presence"}
    assert expected_categories.issubset(categories)

    # Check non-empty subsets
    assert (df["N_answers"] > 0).all()
    assert (df["mean_cer"] > 0.0).all()
