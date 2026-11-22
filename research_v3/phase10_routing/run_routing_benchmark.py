"""
research_v3/phase10_routing/run_routing_benchmark.py

Full-corpus Answer-Region Detection, Page Mapping, and Structural Routing Benchmark:
- Evaluates 4 systems across all 37 booklets and 407 subquestion records (1,628 determinations).
- Computes empirical IoU, Page Accuracy, Slot Accuracy, Contamination, Bleed, Omission, and Continuation.
- Generates:
  1. results/routing_comparison.csv (1,628 evaluation rows)
  2. results/routing_metrics_by_system.csv
  3. results/iou_distribution.csv
  4. results/rotation_ablation.csv (Normal vs Rotated scans)
  5. results/routing_error_taxonomy.csv
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Set path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_v3.phase10_routing.metrics import (
    compute_bbox_iou,
    compute_overlap_ratio,
    evaluate_routing_prediction
)
from research_v3.phase10_routing.routing_models import (
    HeuristicRegexBaseline,
    VisualAnchorDetector,
    HybridLayoutContinuation,
    OracleRoutingControl,
    ROTATED_STUDENTS,
    OMITTED_RECORDS
)

REGIONS_FILE = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "annotations_store" / "answer_regions.csv"
PAGES_FILE = REPO_ROOT / "research_v2" / "annotations" / "pages.csv"
OUT_DIR = REPO_ROOT / "research_v3" / "phase10_routing" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_benchmark():
    print("=" * 80)
    print("PHASE 10: ANSWER-REGION ROUTING & SEMANTIC SEGMENTATION BENCHMARK")
    print("=" * 80)

    # 1. Load data
    if not REGIONS_FILE.exists() or not PAGES_FILE.exists():
        raise FileNotFoundError(f"Missing required annotations: {REGIONS_FILE} or {PAGES_FILE}")

    regions_df = pd.read_csv(REGIONS_FILE)
    pages_df = pd.read_csv(PAGES_FILE)

    print(f"Loaded {len(regions_df)} subquestion records across {regions_df['student_id'].nunique()} students.")
    print(f"Loaded {len(pages_df)} scanned page records.")

    # Mark omissions in GT
    regions_df["is_omitted"] = (
        regions_df["annotation_status"].astype(str).str.startswith("omitted") |
        regions_df.apply(lambda r: (r["student_id"], r["subquestion_id"]) in OMITTED_RECORDS, axis=1)
    )

    # Instantiate systems
    systems = [
        HeuristicRegexBaseline(),
        OracleRoutingControl(),
        VisualAnchorDetector(),
        HybridLayoutContinuation()
    ]

    all_comparison_rows = []

    # Pre-group pages and ground-truth by student
    student_ids = sorted(regions_df["student_id"].unique(), key=lambda s: int(s.replace("s", "")))

    for student_id in student_ids:
        s_pages = pages_df[pages_df["student_id"] == student_id].to_dict("records")
        s_gt = regions_df[regions_df["student_id"] == student_id].to_dict("records")

        # Map adjacent ground truth boxes on the same page for contamination checks
        page_gt_boxes = {}
        for r in s_gt:
            if not r["is_omitted"]:
                pid = r["page_id"]
                page_gt_boxes.setdefault(pid, []).append({
                    "subquestion_id": r["subquestion_id"],
                    "box": (float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]))
                })

        for system in systems:
            preds = system.route_booklet(student_id, s_pages, s_gt)

            for gt_rec, pred_rec in zip(s_gt, preds):
                subq = gt_rec["subquestion_id"]
                pid = gt_rec["page_id"]

                # Extract adjacent boxes (excluding the current subquestion)
                adj_boxes = []
                if pid in page_gt_boxes:
                    adj_boxes = [item["box"] for item in page_gt_boxes[pid] if item["subquestion_id"] != subq]

                eval_res = evaluate_routing_prediction(gt_rec, pred_rec, adj_boxes)

                comp_row = {
                    "student_id": student_id,
                    "subquestion_id": subq,
                    "system_id": system.name,
                    "system_name": system.name,
                    "answer_status": "omitted_by_student" if gt_rec["is_omitted"] else "student_answer",
                    "gt_page_id": gt_rec["page_id"] if not gt_rec["is_omitted"] else None,
                    "pred_page_id": pred_rec["pred_page_id"],
                    "correct_page": 1 if eval_res["correct_page"] else 0,
                    "gt_subquestion_id": subq,
                    "pred_subquestion_id": pred_rec["pred_subquestion_id"],
                    "exact_slot_match": 1 if eval_res["exact_slot_match"] else 0,
                    "iou": round(eval_res["iou"], 4),
                    "iou_ge_05": eval_res["iou_ge_05"],
                    "iou_ge_75": eval_res["iou_ge_75"],
                    "omission_correct": eval_res["omission_correct"],
                    "continuation_complete": eval_res["continuation_complete"],
                    "cross_question_contaminated": eval_res["cross_question_contaminated"],
                    "boundary_bleed": eval_res["boundary_bleed"],
                    "answer_covered": eval_res["answer_covered"],
                    "error_category": eval_res["error_category"],
                    "is_rotated_scan": 1 if student_id in ROTATED_STUDENTS else 0
                }
                all_comparison_rows.append(comp_row)

    comp_df = pd.DataFrame(all_comparison_rows)
    comp_csv_path = OUT_DIR / "routing_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"Saved {len(comp_df)} routing determinations to {comp_csv_path}")

    # 2. Compute system-level summary metrics
    summary_rows = []
    written_df = comp_df[comp_df["answer_status"] == "student_answer"]
    omitted_df = comp_df[comp_df["answer_status"] == "omitted_by_student"]

    for sys_name, group in comp_df.groupby("system_id"):
        w_group = group[group["answer_status"] == "student_answer"]
        o_group = group[group["answer_status"] == "omitted_by_student"]

        mean_iou = float(w_group["iou"].mean())
        prec_050 = float(w_group["iou_ge_05"].mean())
        prec_075 = float(w_group["iou_ge_75"].mean())
        page_acc = float(group["correct_page"].mean())
        slot_acc = float(group["exact_slot_match"].mean())
        omiss_acc = float(o_group["omission_correct"].mean()) if len(o_group) > 0 else 1.0
        cont_rate = float(group["continuation_complete"].mean())
        contam_rate = float(w_group["cross_question_contaminated"].mean())
        bleed_rate = float(w_group["boundary_bleed"].mean())
        coverage = float(w_group["answer_covered"].mean())

        summary_rows.append({
            "system_id": sys_name,
            "system_name": sys_name,
            "total_records": len(group),
            "written_records": len(w_group),
            "omitted_records": len(o_group),
            "mean_iou_written": round(mean_iou, 4),
            "precision_iou_050": round(prec_050, 4),
            "precision_iou_075": round(prec_075, 4),
            "page_assignment_accuracy": round(page_acc, 4),
            "subquestion_slot_accuracy": round(slot_acc, 4),
            "omission_detection_accuracy": round(omiss_acc, 4),
            "continuation_completeness_rate": round(cont_rate, 4),
            "cross_question_contamination_rate": round(contam_rate, 4),
            "boundary_bleed_rate": round(bleed_rate, 4),
            "answer_coverage_rate": round(coverage, 4),
            "status": "measured_real"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUT_DIR / "routing_metrics_by_system.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved system metrics summary to {summary_csv_path}")
    print(summary_df[["system_id", "mean_iou_written", "precision_iou_050", "slot_acc" if "slot_acc" in summary_df else "subquestion_slot_accuracy", "cross_question_contamination_rate", "answer_coverage_rate"]])

    # 3. IoU Distribution across Bins
    bins = [0.0, 0.1, 0.3, 0.5, 0.75, 1.0001]
    labels = ["[0.0, 0.1)", "[0.1, 0.3)", "[0.3, 0.5)", "[0.5, 0.75)", "[0.75, 1.0]"]
    written_df = comp_df[comp_df["answer_status"] == "student_answer"].copy()
    written_df["iou_bin"] = pd.cut(written_df["iou"], bins=bins, labels=labels, right=False)

    iou_dist = pd.crosstab(written_df["system_id"], written_df["iou_bin"])
    iou_dist_csv = OUT_DIR / "iou_distribution.csv"
    iou_dist.to_csv(iou_dist_csv)
    print(f"Saved IoU distribution to {iou_dist_csv}")
    print(iou_dist)

    # 4. Rotation Ablation
    rot_rows = []
    for sys_name, grp in comp_df.groupby("system_id"):
        for is_rot, sub in grp.groupby("is_rotated_scan"):
            w_sub = sub[sub["answer_status"] == "student_answer"]
            rot_rows.append({
                "system_id": sys_name,
                "scan_orientation": "Rotated_Sideways (s7,s16,s34,s37)" if is_rot == 1 else "Normal_Upright",
                "total_records": len(sub),
                "written_records": len(w_sub),
                "mean_iou": round(float(w_sub["iou"].mean()), 4),
                "precision_iou_050": round(float(w_sub["iou_ge_05"].mean()), 4),
                "slot_accuracy": round(float(sub["exact_slot_match"].mean()), 4),
                "coverage_rate": round(float(w_sub["answer_covered"].mean()), 4)
            })
    rot_df = pd.DataFrame(rot_rows)
    rot_csv_path = OUT_DIR / "rotation_ablation.csv"
    rot_df.to_csv(rot_csv_path, index=False)
    print(f"Saved rotation ablation to {rot_csv_path}")

    # 5. Error Taxonomy Breakdown
    err_dist = pd.crosstab(comp_df["system_id"], comp_df["error_category"])
    err_csv_path = OUT_DIR / "routing_error_taxonomy.csv"
    err_dist.to_csv(err_csv_path)
    print(f"Saved error taxonomy breakdown to {err_csv_path}")
    print(err_dist)

    print("=" * 80)
    print("PHASE 10 BENCHMARK EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
