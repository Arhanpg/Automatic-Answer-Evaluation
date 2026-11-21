"""
research_v3/phase10_routing/tests/test_routing.py

Unit tests for Phase 10 Answer-Region Detection, Page Mapping, and Structural Routing:
1. Geometric IoU and Overlap calculations
2. Omission evaluation handling
3. Error taxonomy categorization
4. Rotation angle mappings
5. Routing comparison dataset integrity
"""

import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import pandas as pd
from research_v3.phase10_routing.metrics import (
    compute_bbox_iou,
    compute_overlap_ratio,
    evaluate_routing_prediction
)
from research_v3.phase10_routing.routing_models import (
    ROTATED_STUDENTS,
    OMITTED_RECORDS
)


def test_bbox_iou_geometry():
    # Exact match
    boxA = (100.0, 100.0, 300.0, 300.0)
    assert compute_bbox_iou(boxA, boxA) == 1.0

    # Disjoint boxes
    boxB = (400.0, 400.0, 600.0, 600.0)
    assert compute_bbox_iou(boxA, boxB) == 0.0

    # Partial overlap
    boxC = (200.0, 100.0, 400.0, 300.0)
    # inter: [200, 100, 300, 300] -> w=100, h=200, area=20000
    # areaA: 200*200=40000, areaC: 200*200=40000, union=60000
    # iou = 20000 / 60000 = 1/3
    assert abs(compute_bbox_iou(boxA, boxC) - (1.0 / 3.0)) < 1e-5

    # None or zero-area handling
    assert compute_bbox_iou(None, boxA) == 0.0
    assert compute_bbox_iou(boxA, None) == 0.0
    assert compute_bbox_iou((100.0, 100.0, 100.0, 200.0), boxA) == 0.0


def test_overlap_ratio_calculation():
    target = (100.0, 100.0, 300.0, 300.0)  # area = 40,000
    intruder_half = (200.0, 100.0, 400.0, 300.0)  # overlap = 100 * 200 = 20,000
    assert abs(compute_overlap_ratio(target, intruder_half) - 0.50) < 1e-5

    intruder_full = (50.0, 50.0, 350.0, 350.0)
    assert compute_overlap_ratio(target, intruder_full) == 1.0

    intruder_disjoint = (400.0, 400.0, 500.0, 500.0)
    assert compute_overlap_ratio(target, intruder_disjoint) == 0.0


def test_omission_evaluation():
    gt_omitted = {"subquestion_id": "1b", "is_omitted": True}
    pred_correct_omission = {"pred_subquestion_id": "1b", "is_omitted": True}
    res = evaluate_routing_prediction(gt_omitted, pred_correct_omission)
    assert res["omission_correct"] == 1
    assert res["iou"] == 1.0
    assert res["error_category"] == "none"

    pred_false_positive_written = {
        "pred_page_id": "p1",
        "pred_subquestion_id": "1b",
        "pred_bbox": (100, 100, 200, 200),
        "is_omitted": False
    }
    res = evaluate_routing_prediction(gt_omitted, pred_false_positive_written)
    assert res["omission_correct"] == 0
    assert res["iou"] == 0.0
    assert res["error_category"] == "false_positive_written"

    gt_written = {
        "subquestion_id": "1a",
        "page_id": "p2",
        "x0": 100, "y0": 100, "x1": 300, "y1": 300,
        "is_omitted": False
    }
    pred_false_omission = {"pred_subquestion_id": "1a", "is_omitted": True}
    res = evaluate_routing_prediction(gt_written, pred_false_omission)
    assert res["omission_correct"] == 0
    assert res["iou"] == 0.0
    assert res["error_category"] == "missed_region"


def test_error_categorization():
    gt = {
        "subquestion_id": "1a",
        "page_id": "p2",
        "x0": 100, "y0": 100, "x1": 300, "y1": 300,
        "is_omitted": False
    }

    # Wrong page
    pred_wrong_page = {
        "pred_page_id": "p3",
        "pred_subquestion_id": "1a",
        "pred_bbox": (100, 100, 300, 300),
        "is_omitted": False
    }
    res = evaluate_routing_prediction(gt, pred_wrong_page)
    assert res["correct_page"] is False
    assert res["iou"] == 0.0
    assert res["error_category"] == "wrong_page"

    # Wrong slot
    pred_wrong_slot = {
        "pred_page_id": "p2",
        "pred_subquestion_id": "1b",
        "pred_bbox": (100, 100, 300, 300),
        "is_omitted": False
    }
    res = evaluate_routing_prediction(gt, pred_wrong_slot)
    assert res["exact_slot_match"] is False
    assert res["error_category"] == "wrong_subquestion_header"


def test_rotation_metadata():
    assert "s7" in ROTATED_STUDENTS and ROTATED_STUDENTS["s7"] == 270
    assert "s16" in ROTATED_STUDENTS and ROTATED_STUDENTS["s16"] == 180
    assert "s34" in ROTATED_STUDENTS and ROTATED_STUDENTS["s34"] == 270
    assert "s37" in ROTATED_STUDENTS and ROTATED_STUDENTS["s37"] == 270
    assert len(OMITTED_RECORDS) == 15


def test_routing_benchmark_artifacts_integrity():
    res_dir = REPO_ROOT / "research_v3" / "phase10_routing" / "results"
    comp_file = res_dir / "routing_comparison.csv"
    assert comp_file.exists(), "routing_comparison.csv must exist"

    df = pd.read_csv(comp_file)
    assert len(df) == 1628, f"Expected 1,628 rows (407 x 4), got {len(df)}"
    assert df["student_id"].nunique() == 37

    # Validate Oracle system upper bound
    oracle_df = df[df["system_id"] == "oracle_map_control"]
    written_oracle = oracle_df[oracle_df["answer_status"] == "student_answer"]
    assert written_oracle["iou"].mean() == 1.0
    assert oracle_df["correct_page"].mean() == 1.0
    assert oracle_df["exact_slot_match"].mean() == 1.0
