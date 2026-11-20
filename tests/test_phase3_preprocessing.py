"""
tests/test_phase3_preprocessing.py -- Test Suite for Phase 3 (Gate 3) Preprocessing Ablation & Ink Separation
Verifies:
1. Generation and existence of all Gate 3 artifacts.
2. Complete coverage of 10 paired conditions (P0 through P9).
3. Exact paired design (identical crop keys across all 10 conditions).
4. Physical validity of pixel-level ink removal and preservation metrics (0 <= p <= 1).
5. Statistical significance test integrity (p-values, Wilcoxon, paired t-tests).
6. Split isolation (model selection on development partition only).
"""
import os
import sys
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREP_DIR = ROOT / "research_v2" / "preprocessing"

EXPECTED_CONDITIONS = {"P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"}
DEV_STUDENTS = {"s2", "s5", "s12", "s18", "s21", "s33"}

def test_preprocessing_artifacts_exist():
    """Verify that all four required Phase 3 CSV artifacts exist and are non-empty."""
    required = [
        PREP_DIR / "preprocessing_ablation.csv",
        PREP_DIR / "ablation_summary.csv",
        PREP_DIR / "ink_separation_metrics.csv",
        PREP_DIR / "paired_statistical_tests.csv"
    ]
    for rf in required:
        assert rf.exists(), f"Missing required Phase 3 artifact: {rf}"
        assert rf.stat().st_size > 0, f"Artifact is empty: {rf}"

def test_all_10_conditions_evaluated():
    """Verify that all 10 predeclared conditions (P0-P9) exist in the summary."""
    sum_csv = PREP_DIR / "ablation_summary.csv"
    with open(sum_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    found_cids = {r["condition_id"] for r in rows}
    assert found_cids == EXPECTED_CONDITIONS, f"Conditions mismatch: {found_cids} vs {EXPECTED_CONDITIONS}"

def test_paired_design_consistency():
    """Verify that every evaluated crop has exactly 10 paired condition rows."""
    abl_csv = PREP_DIR / "preprocessing_ablation.csv"
    crop_cond_map = {}
    with open(abl_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["student_id"], r["subquestion_id"])
            if key not in crop_cond_map:
                crop_cond_map[key] = set()
            crop_cond_map[key].add(r["condition_id"])

    assert len(crop_cond_map) == 66, f"Expected 66 development subquestion crops, got {len(crop_cond_map)}"
    for key, cids in crop_cond_map.items():
        assert cids == EXPECTED_CONDITIONS, f"Crop {key} missing conditions: {EXPECTED_CONDITIONS - cids}"

def test_split_isolation():
    """Verify that ablation was strictly performed on the validation development split."""
    abl_csv = PREP_DIR / "preprocessing_ablation.csv"
    with open(abl_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assert r["student_id"] in DEV_STUDENTS, f"Test student leakage in ablation: {r['student_id']}"

def test_ink_separation_metrics_bounds():
    """Verify pixel-level ink separation metrics fall within valid [0.0, 1.0] probability ranges."""
    ink_csv = PREP_DIR / "ink_separation_metrics.csv"
    with open(ink_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 8, f"Expected at least 8 evaluated ink separation crops, got {len(rows)}"
    for r in rows:
        red_rec = float(r["examiner_ink_removal_recall"])
        leakage = float(r["examiner_ink_residual_leakage_rate"])
        stud_rec = float(r["student_ink_preservation_recall"])
        loss = float(r["student_ink_pixel_loss_rate"])

        assert 0.0 <= red_rec <= 1.0, f"Red recall out of bounds: {red_rec}"
        assert 0.0 <= leakage <= 1.0, f"Leakage out of bounds: {leakage}"
        assert 0.0 <= stud_rec <= 1.0, f"Student preservation recall out of bounds: {stud_rec}"
        assert 0.0 <= loss <= 1.0, f"Student loss rate out of bounds: {loss}"

def test_statistical_testing_validity():
    """Verify paired statistical hypothesis testing outputs."""
    stat_csv = PREP_DIR / "paired_statistical_tests.csv"
    with open(stat_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 9, f"Expected 9 pairwise contrasts against P0, got {len(rows)}"
    for r in rows:
        p_val_t = float(r["paired_t_pvalue"])
        p_val_w = float(r["wilcoxon_pvalue"])
        assert 0.0 <= p_val_t <= 1.0, f"Invalid t-test p-value: {p_val_t}"
        assert 0.0 <= p_val_w <= 1.0, f"Invalid Wilcoxon p-value: {p_val_w}"
        assert int(r["n_paired_crops"]) > 0
