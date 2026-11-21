"""
research_v3/phase11_error_propagation/run_error_propagation_benchmark.py

Controlled 2x2 Factorial Error-Propagation Experiment:
- Benchmarks 4 Factorial Streams across 37 student booklets and 148 parent questions.
- Computes Two-Factor Shapley Value Attribution, Interaction Coupling, and Two-Way ANOVA.
- Compares 4 Grading Engines (Unified Hybrid, Dense S-BERT, Symbolic SymPy, Lexical TF-IDF).
- Generates verified research CSV artifacts:
  1. results/stagewise_2x2_predictions.csv
  2. results/stagewise_per_student.csv
  3. results/stagewise_summary.csv
  4. results/anova_decomposition_summary.csv
  5. results/grading_engine_comparison.csv
  6. results/question_level_breakdown.csv
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_v3.phase11_error_propagation.grading_engines import (
    UnifiedHybridGrader,
    DenseSemanticGrader,
    SymbolicMathGrader,
    LexicalRuleGrader,
    PARENT_MAX_MARKS
)
from research_v3.phase11_error_propagation.factorial_streams import (
    FactorialStreamBuilder,
    SUBQ_ORDER
)
from research_v3.phase11_error_propagation.anova_decomposition import (
    compute_evaluation_metrics,
    compute_shapley_decomposition,
    compute_twoway_anova,
    compute_paired_significance
)

MARKS_CSV = REPO_ROOT / "research_v2" / "annotations" / "marks.csv"
OUT_DIR = REPO_ROOT / "research_v3" / "phase11_error_propagation" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth_marks() -> Dict[str, Dict[str, float]]:
    """Loads official teacher marks per student."""
    df = pd.read_csv(MARKS_CSV)
    marks = {}
    for _, row in df.iterrows():
        sid = row["student_id"]
        qid = row["parent_question_id"]
        marks.setdefault(sid, {})[qid] = float(row["teacher_mark"])

    # Compute total
    for sid, q_dict in marks.items():
        q_dict["total"] = sum(q_dict.get(f"Q{i}", 0.0) for i in range(1, 5))
    return marks


def run_benchmark():
    print("=" * 80)
    print("PHASE 11: CONTROLLED 2x2 FACTORIAL ERROR-PROPAGATION EXPERIMENT")
    print("=" * 80)

    # 1. Load data
    gt_marks = load_ground_truth_marks()
    builder = FactorialStreamBuilder()
    student_ids = sorted(gt_marks.keys(), key=lambda s: int(s.replace("s", "")))
    print(f"Loaded {len(student_ids)} student mark records (148 parent questions).")

    # Initialize Graders
    hybrid_grader = UnifiedHybridGrader()
    sbert_grader = DenseSemanticGrader()
    sympy_grader = SymbolicMathGrader()
    lexical_grader = LexicalRuleGrader()

    # Stream definitions for the 2x2 factorial
    # (Stream ID, Display Name, Build Function)
    stream_defs = [
        ("stream_a", "Stream A (Oracle Upper Bound)", lambda sid: builder.build_stream_a(sid)),
        ("stream_b_trocr", "Stream B (Recognition Only: TrOCR-Small Raw)", lambda sid: builder.build_stream_b(sid, "trocr_raw")),
        ("stream_b_tesseract", "Stream B (Recognition Only: Tesseract Preprocessed)", lambda sid: builder.build_stream_b(sid, "tesseract_preprocessed")),
        ("stream_c", "Stream C (Routing Only: Baseline Regex)", lambda sid: builder.build_stream_c(sid)),
        ("stream_d_baseline", "Stream D (End-to-End Baseline)", lambda sid: builder.build_stream_d_baseline(sid, "trocr_raw")),
        ("stream_d_hybrid", "Stream D-Hybrid (End-to-End Proposed)", lambda sid: builder.build_stream_d_hybrid(sid, "trocr_raw"))
    ]

    all_pred_rows = []
    student_stream_totals = {s_id: {} for s_id, _, _ in stream_defs}

    # 2. Grade all students under each stream using the Unified Hybrid Grader
    print("Executing live grading inference across all factorial streams...")
    for s_idx, (stream_id, stream_name, build_fn) in enumerate(stream_defs, 1):
        print(f"[{s_idx}/{len(stream_defs)}] Grading {stream_name}...")
        for sid in student_ids:
            answers = build_fn(sid)
            q_scores = hybrid_grader.grade_student(sid, answers)
            student_stream_totals[stream_id][sid] = q_scores["total"]

            # Log prediction records
            for q_key in ["Q1", "Q2", "Q3", "Q4", "total"]:
                pred_val = q_scores[q_key]
                true_val = gt_marks[sid][q_key]
                all_pred_rows.append({
                    "stream_id": stream_id,
                    "stream_name": stream_name,
                    "student_id": sid,
                    "question_id": q_key,
                    "predicted_mark": pred_val,
                    "teacher_mark": true_val,
                    "absolute_error": abs(pred_val - true_val),
                    "maximum_mark": PARENT_MAX_MARKS[q_key]
                })

    preds_df = pd.DataFrame(all_pred_rows)
    preds_csv_path = OUT_DIR / "stagewise_2x2_predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"Saved {len(preds_df)} stagewise predictions to {preds_csv_path}")

    # 3. Student-level summary
    student_rows = []
    for sid in student_ids:
        row = {"student_id": sid, "teacher_total": gt_marks[sid]["total"]}
        for stream_id, _, _ in stream_defs:
            pred_tot = student_stream_totals[stream_id][sid]
            row[f"{stream_id}_pred"] = pred_tot
            row[f"{stream_id}_abs_err"] = abs(pred_tot - gt_marks[sid]["total"])
        student_rows.append(row)
    student_df = pd.DataFrame(student_rows)
    student_csv_path = OUT_DIR / "stagewise_per_student.csv"
    student_df.to_csv(student_csv_path, index=False)
    print(f"Saved student-level totals to {student_csv_path}")

    # 4. Compute Stream-Level Aggregate Summary & Shapley Attribution
    summary_rows = []
    y_true_totals = np.array([gt_marks[sid]["total"] for sid in student_ids])

    stream_metrics = {}
    for stream_id, stream_name, _ in stream_defs:
        y_pred_totals = np.array([student_stream_totals[stream_id][sid] for sid in student_ids])
        met = compute_evaluation_metrics(y_true_totals, y_pred_totals, max_mark=20.0)
        stream_metrics[stream_id] = met

    # Compute Shapley Attribution for Primary Comparison (A, B-TrOCR, C, D-Baseline)
    mae_a = stream_metrics["stream_a"]["mae"]
    mae_b = stream_metrics["stream_b_trocr"]["mae"]
    mae_c = stream_metrics["stream_c"]["mae"]
    mae_d = stream_metrics["stream_d_baseline"]["mae"]
    shapley = compute_shapley_decomposition(mae_a, mae_b, mae_c, mae_d)

    print("\n--- SHAPLEY ATTRIBUTION ---")
    print(f"Delta Total (D - A): {shapley['delta_total']:.4f}")
    print(f"Delta Recognition (B - A): {shapley['delta_recognition']:.4f}")
    print(f"Delta Routing (C - A): {shapley['delta_routing']:.4f}")
    print(f"Non-linear Interaction (I): {shapley['interaction_coupling']:.4f}")
    print(f"Shapley Recognition: {shapley['shapley_phi_recognition']:.4f} ({shapley['shapley_pct_recognition']:.1f}%)")
    print(f"Shapley Routing: {shapley['shapley_phi_routing']:.4f} ({shapley['shapley_pct_routing']:.1f}%)")

    # Build summary rows with statistical tests against Stream A
    y_pred_a = np.array([student_stream_totals["stream_a"][sid] for sid in student_ids])
    for stream_id, stream_name, _ in stream_defs:
        met = stream_metrics[stream_id]
        y_pred_s = np.array([student_stream_totals[stream_id][sid] for sid in student_ids])

        if stream_id == "stream_a":
            p_val = 1.0
            t_val = 0.0
            cohen_d = 0.0
            shap_attr = "0.0%"
        else:
            sig = compute_paired_significance(y_pred_a, y_pred_s)
            p_val = sig["wilcoxon_p"]
            t_val = sig["paired_t_stat"]
            cohen_d = sig["cohen_d"]
            if stream_id == "stream_b_trocr":
                shap_attr = f"{shapley['shapley_pct_recognition']:.1f}%"
            elif stream_id == "stream_c":
                shap_attr = f"{shapley['shapley_pct_routing']:.1f}%"
            else:
                shap_attr = "-"

        summary_rows.append({
            "stream_id": stream_id,
            "description": stream_name,
            "total_mae": met["mae"],
            "total_rmse": met["rmse"],
            "qwk": met["qwk"],
            "spearman_rho": met["spearman_rho"],
            "within_05m_pct": met["within_05m_pct"],
            "within_10m_pct": met["within_10m_pct"],
            "delta_mae_vs_oracle": round(met["mae"] - mae_a, 4),
            "shapley_attribution": shap_attr,
            "cohen_d": cohen_d,
            "wilcoxon_p": p_val
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUT_DIR / "stagewise_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved stagewise summary to {summary_csv_path}")
    print(summary_df[["stream_id", "total_mae", "total_rmse", "qwk", "delta_mae_vs_oracle", "shapley_attribution", "wilcoxon_p"]])

    # 5. Formal Two-Way ANOVA
    print("\nExecuting Two-Way Factorial ANOVA (Recognition x Routing)...")
    scores_a = np.array([student_stream_totals["stream_a"][sid] for sid in student_ids])
    scores_b = np.array([student_stream_totals["stream_b_trocr"][sid] for sid in student_ids])
    scores_c = np.array([student_stream_totals["stream_c"][sid] for sid in student_ids])
    scores_d = np.array([student_stream_totals["stream_d_baseline"][sid] for sid in student_ids])

    anova_df = compute_twoway_anova(scores_a, scores_b, scores_c, scores_d)
    anova_csv_path = OUT_DIR / "anova_decomposition_summary.csv"
    anova_df.to_csv(anova_csv_path, index=False)
    print(f"Saved ANOVA summary to {anova_csv_path}")
    print(anova_df.to_string())

    # 6. Question-Level Performance Breakdown across Streams
    q_breakdown_rows = []
    for q_key in ["Q1", "Q2", "Q3", "Q4"]:
        for stream_id, stream_name, _ in stream_defs:
            sub_preds = preds_df[(preds_df["stream_id"] == stream_id) & (preds_df["question_id"] == q_key)]
            y_t = sub_preds["teacher_mark"].values
            y_p = sub_preds["predicted_mark"].values
            q_met = compute_evaluation_metrics(y_t, y_p, max_mark=PARENT_MAX_MARKS[q_key])
            q_breakdown_rows.append({
                "question_id": q_key,
                "stream_id": stream_id,
                "stream_name": stream_name,
                "max_mark": PARENT_MAX_MARKS[q_key],
                "mae": q_met["mae"],
                "rmse": q_met["rmse"],
                "within_05m_pct": q_met["within_05m_pct"],
                "within_10m_pct": q_met["within_10m_pct"],
                "qwk": q_met["qwk"],
                "spearman_rho": q_met["spearman_rho"]
            })
    q_breakdown_df = pd.DataFrame(q_breakdown_rows)
    q_csv_path = OUT_DIR / "question_level_breakdown.csv"
    q_breakdown_df.to_csv(q_csv_path, index=False)
    print(f"Saved question-level breakdown to {q_csv_path}")

    # 7. Grading Engine Benchmark (Experiment E6) under Stream A
    print("\nBenchmarking Alternative Grading Engines under Stream A...")
    engine_rows = []
    engines = [
        ("Unified Rubric Hybrid Grader (Ours)", hybrid_grader),
        ("Dense Semantic S-BERT Baseline", sbert_grader),
        ("Symbolic SymPy Math Verifier Baseline", sympy_grader),
        ("Lexical TF-IDF & Keyword Rule Baseline", lexical_grader)
    ]

    for eng_name, eng in engines:
        eng_preds = []
        for sid in student_ids:
            answers = builder.build_stream_a(sid)
            q_res = eng.grade_student(sid, answers)
            eng_preds.append(q_res["total"])
        eng_met = compute_evaluation_metrics(y_true_totals, np.array(eng_preds), max_mark=20.0)
        engine_rows.append({
            "engine_name": eng_name,
            "total_mae": eng_met["mae"],
            "total_rmse": eng_met["rmse"],
            "exact_match_pct": eng_met["exact_match_pct"],
            "within_05m_pct": eng_met["within_05m_pct"],
            "within_10m_pct": eng_met["within_10m_pct"],
            "qwk": eng_met["qwk"],
            "spearman_rho": eng_met["spearman_rho"],
            "pearson_r": eng_met["pearson_r"]
        })

    eng_df = pd.DataFrame(engine_rows)
    eng_csv_path = OUT_DIR / "grading_engine_comparison.csv"
    eng_df.to_csv(eng_csv_path, index=False)
    print(f"Saved grading engine comparison to {eng_csv_path}")
    print(eng_df[["engine_name", "total_mae", "total_rmse", "within_05m_pct", "qwk", "spearman_rho"]])

    print("=" * 80)
    print("PHASE 11 BENCHMARK EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
