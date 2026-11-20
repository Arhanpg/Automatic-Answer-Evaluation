"""
tests/test_phase4_routing.py -- Test Suite for Phase 4 (Gate 4) Layout Analysis & Routing Benchmark
Verifies:
1. Existence and integrity of all Gate 4 routing artifacts.
2. Complete coverage of 4 comparative systems across all 407 records (1,628 rows).
3. Mathematical correctness of Oracle control (IoU = 1.0, Page Acc = 1.0, Slot Acc = 1.0).
4. Physical validity of IoU, precision, recall, and contamination rates (0 <= m <= 1).
5. Error taxonomy compliance with Section 9.3 required tags.
"""
import os
import sys
import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROUTING_DIR = ROOT / "research_v2" / "routing"

EXPECTED_SYSTEMS = {
    "sheet_parser_baseline",
    "oracle_map_control",
    "robust_header_detector",
    "hybrid_layout_continuation"
}

VALID_ERROR_TAGS = {
    "none",
    "missed_region",
    "wrong_page",
    "wrong_subquestion_header",
    "cross_question_merge",
    "answer_split_or_truncated",
    "continuation_not_joined",
    "omission_false_positive",
    "omission_false_negative",
    "teacher_ink_as_student_text",
    "student_text_as_teacher_ink",
    "ambiguous_or_unresolvable"
}

def test_routing_artifacts_exist():
    """Verify that all four required Phase 4 CSV artifacts exist and are populated."""
    required = [
        ROUTING_DIR / "routing_comparison.csv",
        ROUTING_DIR / "routing_metrics_by_system.csv",
        ROUTING_DIR / "routing_errors.csv",
        ROUTING_DIR / "iou_distribution.csv"
    ]
    for rf in required:
        assert rf.exists(), f"Missing required Phase 4 artifact: {rf}"
        assert rf.stat().st_size > 0, f"Artifact is empty: {rf}"

def test_complete_407_records_per_system():
    """Verify that all 407 records are evaluated across each of the 4 systems (1,628 total)."""
    comp_csv = ROUTING_DIR / "routing_comparison.csv"
    with open(comp_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 407 * 4, f"Expected 1628 rows, got {len(rows)}"
    sys_counts = {}
    for r in rows:
        sys_counts[r["system_id"]] = sys_counts.get(r["system_id"], 0) + 1

    assert set(sys_counts.keys()) == EXPECTED_SYSTEMS
    for sid, count in sys_counts.items():
        assert count == 407, f"System {sid} has {count} rows instead of 407"

def test_oracle_control_invariant():
    """Verify that the Oracle upper bound control achieves 100% precision, IoU, and accuracy."""
    sum_csv = ROUTING_DIR / "routing_metrics_by_system.csv"
    with open(sum_csv, "r", encoding="utf-8") as f:
        rows = {r["system_id"]: r for r in csv.DictReader(f)}

    oracle = rows["oracle_map_control"]
    assert float(oracle["mean_iou_written"]) == 1.0
    assert float(oracle["precision_iou_050"]) == 1.0
    assert float(oracle["page_assignment_accuracy"]) == 1.0
    assert float(oracle["subquestion_slot_accuracy"]) == 1.0
    assert float(oracle["cross_question_contamination_rate"]) == 0.0

def test_metric_bounds_and_coverage():
    """Verify that all system metrics fall within [0.0, 1.0]."""
    sum_csv = ROUTING_DIR / "routing_metrics_by_system.csv"
    with open(sum_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for field in [
                "mean_iou_written", "precision_iou_050", "precision_iou_075",
                "page_assignment_accuracy", "subquestion_slot_accuracy",
                "omission_detection_accuracy", "continuation_completeness_rate",
                "cross_question_contamination_rate", "answer_coverage_rate"
            ]:
                val = float(r[field])
                assert 0.0 <= val <= 1.0, f"Field {field} out of bounds ({val}) for {r['system_id']}"

def test_error_taxonomy_tags():
    """Verify that all recorded routing errors belong to the predeclared taxonomy."""
    err_csv = ROUTING_DIR / "routing_errors.csv"
    with open(err_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cat = r["error_category"]
            assert cat in VALID_ERROR_TAGS, f"Unknown error category: {cat}"
