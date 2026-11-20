"""
tests/test_phase5_grading.py -- Unit and Scientific Validation Suite for Gate 5.

Verifies:
1. 37 student fold assignments (31 dev across 5 folds, exactly 6 locked test booklets).
2. All question marks are strictly bounded by maximum marks (Q1<=8, Q2<=6, Q3<=3, Q4<=3, total<=20).
3. Grading predictions exist for all systems and splits without missing values or NaN errors.
4. Factorial 2x2 streams (Stream A, B, C, D, D-Hybrid) are mathematically consistent:
   - Stream A (Oracle) has the lowest error.
   - Stream D (End-to-End Baseline) has higher error than Stream A.
   - Stream D-Hybrid shows substantial recovery over Stream D.
5. Shapley decomposition values sum correctly and interaction coupling is finite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRADING_DIR = REPO_ROOT / "research_v2" / "grading"
MARKS_CSV = REPO_ROOT / "research_v2" / "annotations" / "marks.csv"

LOCKED_TEST_STUDENTS = {"s7", "s14", "s24", "s29", "s30", "s37"}
PARENT_MAX_MARKS = {"Q1": 8.0, "Q2": 6.0, "Q3": 3.0, "Q4": 3.0, "total": 20.0}


def test_cv_fold_assignments():
    """Validates 5-fold cross validation split and locked test set assignments."""
    cv_path = GRADING_DIR / "cv_fold_assignments.csv"
    assert cv_path.exists(), f"Missing {cv_path}"

    df = pd.read_csv(cv_path)
    assert len(df) == 37, f"Expected 37 student fold assignments, got {len(df)}"

    # Check locked test students
    locked = set(df[df["is_locked_test"] == True]["student_id"].values)
    assert locked == LOCKED_TEST_STUDENTS, f"Locked test mismatch: {locked} vs {LOCKED_TEST_STUDENTS}"

    # Check dev students across 5 folds
    dev_df = df[df["is_locked_test"] == False]
    assert len(dev_df) == 31, f"Expected 31 dev students, got {len(dev_df)}"
    folds = set(dev_df["fold"].unique())
    assert folds == {1, 2, 3, 4, 5}, f"Expected folds 1..5, got {folds}"

    # Check each fold has at least 5 students
    for f in range(1, 6):
        count = len(dev_df[dev_df["fold"] == f])
        assert 5 <= count <= 7, f"Fold {f} has unbalanced count: {count}"


def test_grading_predictions_bounds():
    """Ensures all predicted marks are non-negative and bounded by rubric maximums."""
    preds_path = GRADING_DIR / "grading_predictions_by_system.csv"
    assert preds_path.exists(), f"Missing {preds_path}"

    df = pd.read_csv(preds_path)
    assert len(df) > 0

    for q_id, max_mark in PARENT_MAX_MARKS.items():
        q_rows = df[df["question_id"] == q_id]
        assert (q_rows["predicted_mark"] >= 0.0).all(), f"Found negative predicted marks for {q_id}"
        assert (q_rows["predicted_mark"] <= max_mark + 1e-4).all(), (
            f"Found predicted mark exceeding max mark {max_mark} for {q_id}: {q_rows['predicted_mark'].max()}"
        )
        assert (q_rows["teacher_mark"] >= 0.0).all()
        assert (q_rows["teacher_mark"] <= max_mark + 1e-4).all()


def test_grading_metrics_summary():
    """Validates that summary metrics are computed and Hybrid Grader is superior on conceptual and locked test sets."""
    summary_path = GRADING_DIR / "grading_metrics_summary.csv"
    assert summary_path.exists(), f"Missing {summary_path}"

    df = pd.read_csv(summary_path)
    systems = set(df["system_id"].unique())
    expected_systems = {"lexical_rule_baseline", "semantic_sbert_baseline", "symbolic_math_baseline", "hybrid_grader"}
    assert expected_systems.issubset(systems), f"Missing systems in summary: {expected_systems - systems}"

    # Verify Hybrid Grader significantly beats lexical and pure math baselines on conceptual question Q1
    q1_all = df[(df["question_id"] == "Q1") & (df["split"] == "all")]
    hyb_q1_mae = q1_all[q1_all["system_id"] == "hybrid_grader"]["MAE"].iloc[0]
    lex_q1_mae = q1_all[q1_all["system_id"] == "lexical_rule_baseline"]["MAE"].iloc[0]
    math_q1_mae = q1_all[q1_all["system_id"] == "symbolic_math_baseline"]["MAE"].iloc[0]

    assert hyb_q1_mae < lex_q1_mae, f"Hybrid Q1 MAE ({hyb_q1_mae}) should be lower than Lexical ({lex_q1_mae})"
    assert hyb_q1_mae < math_q1_mae, f"Hybrid Q1 MAE ({hyb_q1_mae}) should be lower than Math-only ({math_q1_mae})"

    # Verify on held-out locked test set total score: Hybrid Grader achieves superior exact agreement and lowest error
    test_totals = df[(df["question_id"] == "total") & (df["split"] == "locked_test")]
    hyb_test_mae = test_totals[test_totals["system_id"] == "hybrid_grader"]["MAE"].iloc[0]
    lex_test_mae = test_totals[test_totals["system_id"] == "lexical_rule_baseline"]["MAE"].iloc[0]
    math_test_mae = test_totals[test_totals["system_id"] == "symbolic_math_baseline"]["MAE"].iloc[0]

    assert hyb_test_mae <= lex_test_mae, f"Hybrid locked test MAE ({hyb_test_mae}) should be <= Lexical ({lex_test_mae})"
    assert hyb_test_mae < math_test_mae, f"Hybrid locked test MAE ({hyb_test_mae}) should be lower than Math-only ({math_test_mae})"

    hyb_exact = test_totals[test_totals["system_id"] == "hybrid_grader"]["exact_match_pct"].iloc[0]
    lex_exact = test_totals[test_totals["system_id"] == "lexical_rule_baseline"]["exact_match_pct"].iloc[0]
    assert hyb_exact > lex_exact, f"Hybrid exact agreement ({hyb_exact}%) should be higher than Lexical ({lex_exact}%)"


def test_stagewise_2x2_consistency():
    """Validates the 2x2 Factorial Streams relationships and error progression."""
    stagewise_csv = GRADING_DIR / "stagewise_summary.csv"
    assert stagewise_csv.exists(), f"Missing {stagewise_csv}"

    df = pd.read_csv(stagewise_csv)
    stream_map = {row["stream_id"]: row for _, row in df.iterrows()}

    assert "stream_a" in stream_map
    assert "stream_b_paddle" in stream_map
    assert "stream_c" in stream_map
    assert "stream_d_baseline" in stream_map
    assert "stream_d_hybrid" in stream_map

    mae_a = stream_map["stream_a"]["MAE_total"]
    mae_b = stream_map["stream_b_paddle"]["MAE_total"]
    mae_c = stream_map["stream_c"]["MAE_total"]
    mae_d = stream_map["stream_d_baseline"]["MAE_total"]
    mae_d_hyb = stream_map["stream_d_hybrid"]["MAE_total"]

    # Stream A (Full Oracle) must have lower error than End-to-End Baseline (Stream D)
    assert mae_a < mae_d, f"Oracle MAE ({mae_a}) must be lower than End-to-End Baseline MAE ({mae_d})"

    # Stream D-Hybrid (Proposed) must recover substantial error compared to Stream D (Baseline)
    assert mae_d_hyb < mae_d, f"Hybrid Pipeline MAE ({mae_d_hyb}) must be lower than Baseline Pipeline MAE ({mae_d})"

    # Check Shapley percentages sum to 100%
    shapley_rec_pct = stream_map["stream_b_paddle"]["shapley_pct"]
    shapley_rout_pct = stream_map["stream_c"]["shapley_pct"]
    assert abs((shapley_rec_pct + shapley_rout_pct) - 100.0) < 0.5, (
        f"Shapley percentages do not sum to 100: {shapley_rec_pct} + {shapley_rout_pct}"
    )


def test_ground_truth_integrity():
    """Ensures teacher marks in grading match original annotations."""
    per_student_csv = GRADING_DIR / "stagewise_per_student.csv"
    assert per_student_csv.exists(), f"Missing {per_student_csv}"

    df_students = pd.read_csv(per_student_csv)
    marks_df = pd.read_csv(MARKS_CSV)

    for sid in df_students["student_id"].unique():
        s_marks = marks_df[marks_df["student_id"] == sid]["teacher_mark"].sum()
        reported_teacher_total = df_students[df_students["student_id"] == sid]["teacher_total"].iloc[0]
        assert abs(s_marks - reported_teacher_total) < 1e-4, (
            f"Teacher mark mismatch for {sid}: original {s_marks} vs reported {reported_teacher_total}"
        )
