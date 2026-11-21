"""
research_v3/phase9_ocr_benchmarking/statistical_tests.py

Rigorous statistical significance testing for handwriting preprocessing ablation:
1. Paired Student's t-test (parametric)
2. Wilcoxon signed-rank test (non-parametric, robust to skewed edit distance distributions)
3. Effect size: Cohen's d for paired samples
4. Bootstrap 95% confidence intervals for CER and WER
Resolves the historical anomaly where t-test was p=0.0327 but Wilcoxon was p=0.1689.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE9_DIR = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking"
RESULTS_DIR = PHASE9_DIR / "results"
METRICS_CSV = RESULTS_DIR / "ocr_metrics_by_answer.csv"
OUTPUT_CSV = RESULTS_DIR / "paired_statistical_tests.csv"


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes two-sided bootstrap confidence interval."""
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Cohen's d for paired samples: mean(diff) / std(diff)."""
    diff = x - y
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def run_tests():
    print("=" * 80)
    print("PHASE 9: STATISTICAL SIGNIFICANCE TESTS (RAW VS PREPROCESSED)")
    print("=" * 80)

    if not METRICS_CSV.exists():
        print(f"Error: {METRICS_CSV} missing. Run run_benchmark.py first.")
        sys.exit(1)

    df = pd.read_csv(METRICS_CSV)
    # Filter to written items (omitted items have CER=0 for both, adding zero variance)
    df = df[~df["is_omitted"]]

    engines = df["engine"].unique()
    test_rows = []

    for eng in engines:
        eng_df = df[df["engine"] == eng]
        raw_df = eng_df[eng_df["condition"] == "raw"].sort_values(["student_id", "subquestion_id"])
        prep_df = eng_df[eng_df["condition"] == "preprocessed"].sort_values(["student_id", "subquestion_id"])

        if len(raw_df) != len(prep_df):
            print(f"Warning: Record count mismatch for {eng}: {len(raw_df)} raw vs {len(prep_df)} preprocessed")
            continue

        for metric in ["cer", "wer", "math_f1", "num_f1"]:
            raw_vals = raw_df[metric].values
            prep_vals = prep_df[metric].values

            mean_raw = float(np.mean(raw_vals))
            mean_prep = float(np.mean(prep_vals))
            delta = mean_prep - mean_raw

            raw_ci = bootstrap_ci(raw_vals)
            prep_ci = bootstrap_ci(prep_vals)

            # Paired t-test
            t_res = stats.ttest_rel(prep_vals, raw_vals)
            t_stat = float(t_res.statistic)
            t_pval = float(t_res.pvalue)

            # Wilcoxon signed-rank test
            diffs = prep_vals - raw_vals
            if np.all(diffs == 0):
                w_stat, w_pval = 0.0, 1.0
            else:
                try:
                    w_res = stats.wilcoxon(prep_vals, raw_vals)
                    w_stat = float(w_res.statistic)
                    w_pval = float(w_res.pvalue)
                except Exception:
                    w_stat, w_pval = 0.0, 1.0

            d_val = compute_cohens_d(prep_vals, raw_vals)

            test_rows.append({
                "engine": eng,
                "metric": metric,
                "sample_size": len(raw_vals),
                "mean_raw": round(mean_raw, 4),
                "raw_ci_95": f"[{raw_ci[0]:.4f}, {raw_ci[1]:.4f}]",
                "mean_preprocessed": round(mean_prep, 4),
                "prep_ci_95": f"[{prep_ci[0]:.4f}, {prep_ci[1]:.4f}]",
                "delta": round(delta, 4),
                "cohens_d": round(d_val, 4),
                "paired_t_stat": round(t_stat, 4),
                "paired_t_pval": round(t_pval, 5),
                "wilcoxon_stat": round(w_stat, 4),
                "wilcoxon_pval": round(w_pval, 5),
                "statistically_significant_05": (t_pval < 0.05 and w_pval < 0.05)
            })

    results_df = pd.DataFrame(test_rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPaired statistical test results saved to: {OUTPUT_CSV}")

    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL COMPARISONS SUMMARY:")
    print("=" * 80)
    for _, r in results_df.iterrows():
        sig_str = "YES (Significant)" if r["statistically_significant_05"] else "NO (Not Significant)"
        print(f"  [{r['engine'].upper()} - {r['metric'].upper()}] Raw: {r['mean_raw']:.4f} -> Prep: {r['mean_preprocessed']:.4f} (Delta: {r['delta']:+.4f}) | t-pval: {r['paired_t_pval']:.5f}, w-pval: {r['wilcoxon_pval']:.5f} | Sig? {sig_str}")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
