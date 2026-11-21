"""
research_v3/phase9_ocr_benchmarking/subgroup_analysis.py

Stratified subgroup evaluation across question modalities:
1. Mathematical and Algorithmic Derivations: Q2a, Q2b, Q2c, Q4
   (Involves vectors, dot products, softmax fractions, Laplace smoothing ratios, tokens <s>, </s>)
2. Conceptual Prose Answers: Q1a, Q1b, Q1c, Q1d, Q3a, Q3b, Q3c
   (Involves natural language, transformer architectural definitions, decoding concepts)

Quantifies the mathematical recognition gap: why technical examination handwriting recognition
degrades substantially on symbols, equations, and multi-line arithmetic.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE9_DIR = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking"
RESULTS_DIR = PHASE9_DIR / "results"
METRICS_CSV = RESULTS_DIR / "ocr_metrics_by_answer.csv"
OUTPUT_CSV = RESULTS_DIR / "subgroup_analysis.csv"

MATH_SUBQUESTIONS = {"2a", "2b", "2c", "4"}
PROSE_SUBQUESTIONS = {"1a", "1b", "1c", "1d", "3a", "3b", "3c"}


def run_subgroup_analysis():
    print("=" * 80)
    print("PHASE 9: SUBGROUP ANALYSIS (MATHEMATICAL DERIVATIONS VS CONCEPTUAL PROSE)")
    print("=" * 80)

    if not METRICS_CSV.exists():
        print(f"Error: {METRICS_CSV} missing. Run run_benchmark.py first.")
        sys.exit(1)

    df = pd.read_csv(METRICS_CSV)
    # Focus on written answers
    df = df[~df["is_omitted"]]

    subgroup_rows = []

    for eng in df["engine"].unique():
        for cond in df["condition"].unique():
            sub = df[(df["engine"] == eng) & (df["condition"] == cond)]

            math_sub = sub[sub["subquestion_id"].isin(MATH_SUBQUESTIONS)]
            prose_sub = sub[sub["subquestion_id"].isin(PROSE_SUBQUESTIONS)]

            m_cer = float(math_sub["cer"].mean())
            p_cer = float(prose_sub["cer"].mean())
            gap_cer = m_cer - p_cer

            m_wer = float(math_sub["wer"].mean())
            p_wer = float(prose_sub["wer"].mean())
            gap_wer = m_wer - p_wer

            subgroup_rows.append({
                "engine": eng,
                "condition": cond,
                "math_sample_size": len(math_sub),
                "prose_sample_size": len(prose_sub),
                "math_mean_cer": round(m_cer, 4),
                "prose_mean_cer": round(p_cer, 4),
                "cer_gap_math_minus_prose": round(gap_cer, 4),
                "math_mean_wer": round(m_wer, 4),
                "prose_mean_wer": round(p_wer, 4),
                "wer_gap_math_minus_prose": round(gap_wer, 4),
                "math_symbol_f1": round(math_sub["math_f1"].mean(), 4),
                "prose_symbol_f1": round(prose_sub["math_f1"].mean(), 4),
                "math_numeric_f1": round(math_sub["num_f1"].mean(), 4),
                "prose_numeric_f1": round(prose_sub["num_f1"].mean(), 4),
            })

    res_df = pd.DataFrame(subgroup_rows)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubgroup analysis results saved to: {OUTPUT_CSV}")

    print("\n" + "=" * 80)
    print("SUBGROUP COMPARISON (MATH DERIVATIONS VS CONCEPTUAL PROSE):")
    print("=" * 80)
    print(res_df[["engine", "condition", "math_mean_cer", "prose_mean_cer", "cer_gap_math_minus_prose", "math_symbol_f1", "math_numeric_f1"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_subgroup_analysis()
