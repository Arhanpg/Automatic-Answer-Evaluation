"""
research_v3/phase12_risk_control/run_risk_control_benchmark.py

Input-Side Calibrated Risk Control & Selective Human Review Benchmark:
- Completely eliminates gold teacher mark leakage (remedying Defect G).
- Extracts input-side perception, syntax, and routing uncertainty metrics.
- Benchmarks 5 selective review strategies across 12 review budgets.
- Produces:
  1. results/selective_review_curves.csv
  2. results/calibration_correlation.csv
  3. results/student_uncertainty_scores.csv
  4. results/subgroup_risk_profile.csv
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_v3.phase12_risk_control.uncertainty_estimator import (
    InputSideUncertaintyEstimator,
    compute_normalized_edit_distance
)
from research_v3.phase12_risk_control.selective_review import (
    simulate_selective_review,
    simulate_random_selective_review
)

OUT_DIR = REPO_ROOT / "research_v3" / "phase12_risk_control" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKS_CSV = REPO_ROOT / "research_v2" / "annotations" / "marks.csv"
OCR_JSONL = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking" / "results" / "predictions.jsonl"
ROUTING_CSV = REPO_ROOT / "research_v3" / "phase10_routing" / "results" / "routing_comparison.csv"
STAGEWISE_CSV = REPO_ROOT / "research_v3" / "phase11_error_propagation" / "results" / "stagewise_per_student.csv"

SUBQ_ORDER = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b", "3c", "4"]


def run_benchmark():
    print("=" * 80)
    print("PHASE 12: INPUT-SIDE CALIBRATED RISK CONTROL & SELECTIVE REVIEW")
    print("=" * 80)

    # 1. Load data
    stagewise_df = pd.read_csv(STAGEWISE_CSV)
    routing_df = pd.read_csv(ROUTING_CSV)

    # Load OCR predictions
    ocr_trocr = {}
    ocr_tess = {}
    with open(OCR_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            eng = rec.get("engine")
            cond = rec.get("condition")
            sid = rec["student_id"]
            subq = rec["subquestion_id"]
            text = rec.get("hypothesis", "")

            if eng == "trocr" and cond == "raw":
                ocr_trocr.setdefault(sid, {})[subq] = text
            elif eng == "tesseract" and cond == "preprocessed":
                ocr_tess.setdefault(sid, {})[subq] = text

    # Group routing info by student
    student_routing = {}
    for _, row in routing_df[routing_df["system_id"] == "sheet_parser_baseline"].iterrows():
        sid = row["student_id"]
        subq = row["subquestion_id"]
        student_routing.setdefault(sid, {})[subq] = {
            "iou": float(row["iou"]),
            "error_category": row["error_category"],
            "correct_page": bool(row["correct_page"]),
            "exact_slot_match": bool(row["exact_slot_match"])
        }

    # 2. Extract input-side uncertainty scores
    estimator = InputSideUncertaintyEstimator()
    student_records = []
    unc_rows = []

    print("Computing strictly input-side uncertainty metrics (zero label leakage)...")
    for _, s_row in stagewise_df.iterrows():
        sid = s_row["student_id"]
        y_true = float(s_row["teacher_total"])
        y_pred = float(s_row["stream_d_baseline_pred"])
        abs_err = abs(y_pred - y_true)

        s_trocr = ocr_trocr.get(sid, {})
        s_tess = ocr_tess.get(sid, {})
        s_rout = student_routing.get(sid, {})

        unc_data = estimator.estimate_student_uncertainty(sid, s_trocr, s_tess, s_rout)
        comp_unc = unc_data["composite_uncertainty"]
        ocr_dis = unc_data["ocr_disagreement"]
        rout_unc = unc_data["routing_uncertainty"]
        math_unc = unc_data["math_uncertainty"]

        unc_rows.append({
            "student_id": sid,
            "teacher_total": y_true,
            "predicted_total": y_pred,
            "absolute_error": round(abs_err, 4),
            "composite_uncertainty": comp_unc,
            "ocr_disagreement": ocr_dis,
            "routing_uncertainty": rout_unc,
            "math_uncertainty": math_unc
        })

        student_records.append({
            "student_id": sid,
            "teacher_total": y_true,
            "predicted_total": y_pred,
            "absolute_error": abs_err,
            "composite_uncertainty": comp_unc,
            "ocr_disagreement": ocr_dis,
            "routing_uncertainty": rout_unc,
            "math_uncertainty": math_unc
        })

    unc_df = pd.DataFrame(unc_rows)
    unc_csv_path = OUT_DIR / "student_uncertainty_scores.csv"
    unc_df.to_csv(unc_csv_path, index=False)
    print(f"Saved {len(unc_df)} student uncertainty profiles to {unc_csv_path}")

    # 3. Calibration Correlation with Actual Error
    corr_rows = []
    abs_errors = unc_df["absolute_error"].values

    for metric_name, col_name in [
        ("Composite Input-Side Uncertainty (Ours)", "composite_uncertainty"),
        ("OCR Multi-Engine Disagreement Only", "ocr_disagreement"),
        ("Routing Ambiguity Only", "routing_uncertainty"),
        ("Mathematical Notation Ambiguity Only", "math_uncertainty")
    ]:
        vals = unc_df[col_name].values
        try:
            spearman_rho, s_p = stats.spearmanr(vals, abs_errors)
            spearman_rho = float(spearman_rho) if not np.isnan(spearman_rho) else 0.0
        except Exception:
            spearman_rho, s_p = 0.0, 1.0

        try:
            pearson_r, p_p = stats.pearsonr(vals, abs_errors)
            pearson_r = float(pearson_r) if not np.isnan(pearson_r) else 0.0
        except Exception:
            pearson_r, p_p = 0.0, 1.0

        corr_rows.append({
            "uncertainty_estimator": metric_name,
            "spearman_rho": round(spearman_rho, 4),
            "spearman_p": float(s_p),
            "pearson_r": round(pearson_r, 4),
            "pearson_p": float(p_p)
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_csv_path = OUT_DIR / "calibration_correlation.csv"
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"\nSaved calibration correlations to {corr_csv_path}")
    print(corr_df.to_string())

    # 4. Selective Review Curves across Review Budgets
    print("\nSimulating Selective Human Review across budgets...")
    strategies = [
        ("Proposed Input-Side Calibrated Ensemble (Ours)", "composite_uncertainty", True, False),
        ("OCR Multi-Engine Disagreement Only", "ocr_disagreement", True, False),
        ("Routing Ambiguity Only", "routing_uncertainty", True, False),
        ("Oracle Absolute Error Ranking (Upper Bound)", "absolute_error", True, False),
        ("Random Review Baseline (Lower Bound)", None, False, True)
    ]

    all_curve_rows = []
    budget_fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]

    for strat_name, rank_key, rev, is_rand in strategies:
        if is_rand:
            res = simulate_random_selective_review(student_records, n_trials=100, budget_fractions=budget_fractions)
        else:
            res = simulate_selective_review(student_records, rank_key, reverse=rev, budget_fractions=budget_fractions)

        for r in res:
            r["triage_strategy"] = strat_name
            all_curve_rows.append(r)

    curve_df = pd.DataFrame(all_curve_rows)
    curve_csv_path = OUT_DIR / "selective_review_curves.csv"
    curve_df.to_csv(curve_csv_path, index=False)
    print(f"Saved selective review curves to {curve_csv_path}")

    # Print summary of selective review at key operational budgets
    key_budgets = [0.0, 0.10, 0.20, 0.30, 0.50, 1.00]
    pivot_mae = curve_df[curve_df["review_budget_fraction"].isin(key_budgets)].pivot(
        index="triage_strategy", columns="review_budget_pct", values="residual_MAE"
    )
    print("\n--- RESIDUAL MAE ACROSS HUMAN REVIEW BUDGETS ---")
    print(pivot_mae.to_string())

    # 5. Subgroup Risk Stratification
    print("\nAnalyzing Subgroup Risk Stratification...")
    # Classify high vs low risk scripts using the median composite uncertainty
    median_unc = float(unc_df["composite_uncertainty"].median())
    unc_df["risk_stratum"] = unc_df["composite_uncertainty"].apply(
        lambda u: "High Uncertainty (u >= median)" if u >= median_unc else "Low Uncertainty (u < median)"
    )

    subgroup_rows = []
    for stratum, grp in unc_df.groupby("risk_stratum"):
        subgroup_rows.append({
            "subgroup_name": stratum,
            "N_students": len(grp),
            "mean_composite_uncertainty": round(float(grp["composite_uncertainty"].mean()), 4),
            "mean_absolute_error": round(float(grp["absolute_error"].mean()), 4),
            "mean_ocr_disagreement": round(float(grp["ocr_disagreement"].mean()), 4),
            "mean_routing_uncertainty": round(float(grp["routing_uncertainty"].mean()), 4)
        })

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_csv = OUT_DIR / "subgroup_risk_profile.csv"
    subgroup_df.to_csv(subgroup_csv, index=False)
    print(f"Saved subgroup risk profile to {subgroup_csv}")
    print(subgroup_df.to_string())

    print("=" * 80)
    print("PHASE 12 BENCHMARK EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
