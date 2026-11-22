"""
research_v4/model_comparison/src/reporting/plot_figures.py

Generates publication-quality figures for Paper 2:
1. Overall Model Performance (MAE and QWK with 95% CIs).
2. Per-Question Performance Breakdown (Q1 Prose, Q2 Math, Q3 Algos, Q4 Stats).
3. Feature Family Ablation Impact.
4. Input Modality Sensitivity (Clean vs. TrOCR vs. Tesseract).
5. Robustness Noise Decay Curves.
6. Computational Pareto Frontier (Accuracy vs. Latency).
"""

import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def parse_ci_string(ci_str: str) -> tuple:
    """Parses '[lower, upper]' into floats."""
    match = re.search(r"\[([0-9\.\-]+),\s*([0-9\.\-]+)\]", str(ci_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def plot_overall_performance():
    csv_path = TABLES_DIR / "main_model_comparison.csv"
    if not csv_path.exists():
        print(f"Skipping plot_overall_performance: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    # Sort by MAE ascending
    df = df.sort_values(by="mae", ascending=True).reset_index(drop=True)

    # Format model display names
    display_names = [m.replace("_", " ").title() for m in df["model_id"]]

    mae_vals = df["mae"].values
    mae_err_low = []
    mae_err_high = []
    for i, row in df.iterrows():
        low, high = parse_ci_string(row.get("mae_95_ci", ""))
        mae_err_low.append(max(0.0, row["mae"] - low))
        mae_err_high.append(max(0.0, high - row["mae"]))

    y_pos = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    # Subplot 1: MAE with CIs
    ax1.errorbar(
        mae_vals, y_pos,
        xerr=[mae_err_low, mae_err_high],
        fmt='o', color="#1f77b4", ecolor="#7fa9cf", elinewidth=2, capsize=3, markersize=5
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean Absolute Error (Marks, lower is better)")
    ax1.set_title("(A) MAE with 95% Clustered Bootstrap CI")
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    # Subplot 2: QWK with CIs
    qwk_vals = df["qwk"].values
    qwk_err_low = []
    qwk_err_high = []
    for i, row in df.iterrows():
        low, high = parse_ci_string(row.get("qwk_95_ci", ""))
        qwk_err_low.append(max(0.0, row["qwk"] - low))
        qwk_err_high.append(max(0.0, high - row["qwk"]))

    ax2.errorbar(
        qwk_vals, y_pos,
        xerr=[qwk_err_low, qwk_err_high],
        fmt='s', color="#2ca02c", ecolor="#8dd38d", elinewidth=2, capsize=3, markersize=5
    )
    ax2.set_xlabel("Quadratic Weighted Kappa (QWK, higher is better)")
    ax2.set_title("(B) QWK with 95% Clustered Bootstrap CI")
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_file = FIGURES_DIR / "overall_model_performance_qwk_mae.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Generated: {out_file}")


def plot_per_question_heatmap():
    csv_path = TABLES_DIR / "model_performance_by_question.csv"
    if not csv_path.exists():
        print(f"Skipping plot_per_question_heatmap: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    # Pivot for MAE
    pivot_mae = df.pivot(index="model_id", columns="question_id", values="mae")
    pivot_qwk = df.pivot(index="model_id", columns="question_id", values="qwk")

    # Order models by average MAE
    model_order = pivot_mae.mean(axis=1).sort_values().index
    pivot_mae = pivot_mae.loc[model_order]
    pivot_qwk = pivot_qwk.loc[model_order]

    q_labels = ["Q1: Prose", "Q2: Math", "Q3: Algos", "Q4: Stats"]
    m_labels = [m.replace("_", " ").title() for m in model_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # Heatmap 1: MAE
    im1 = ax1.imshow(pivot_mae.values, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(len(q_labels)))
    ax1.set_xticklabels(q_labels)
    ax1.set_yticks(range(len(m_labels)))
    ax1.set_yticklabels(m_labels)
    ax1.set_title("Per-Question MAE (Lower is better)")
    plt.colorbar(im1, ax=ax1, shrink=0.7, label="MAE (Marks)")

    # Annotate values in cells
    for i in range(len(m_labels)):
        for j in range(len(q_labels)):
            val = pivot_mae.values[i, j]
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 2.0 else "white", fontsize=8)

    # Heatmap 2: QWK
    im2 = ax2.imshow(pivot_qwk.values, cmap="viridis", aspect="auto")
    ax2.set_xticks(range(len(q_labels)))
    ax2.set_xticklabels(q_labels)
    ax2.set_yticks(range(len(m_labels)))
    ax2.set_yticklabels([])
    ax2.set_title("Per-Question QWK (Higher is better)")
    plt.colorbar(im2, ax=ax2, shrink=0.7, label="QWK")

    for i in range(len(m_labels)):
        for j in range(len(q_labels)):
            val = pivot_qwk.values[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.5 else "black", fontsize=8)

    plt.tight_layout()
    out_file = FIGURES_DIR / "per_question_performance_heatmap.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Generated: {out_file}")


def plot_feature_ablation():
    csv_path = TABLES_DIR / "feature_ablation_summary.csv"
    if not csv_path.exists():
        print(f"Skipping plot_feature_ablation: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values(by="ridge_mae", ascending=True).reset_index(drop=True)

    fam_labels = [f.replace("_", " ").title() for f in df["feature_family_configuration"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df))

    bars = ax.barh(y_pos, df["ridge_mae"], color="#3470a3", edgecolor="black", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fam_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Ridge Out-of-Fold MAE (Marks)")
    ax.set_title("Feature Family Ablation Impact on Authentic Handwritten Text (TrOCR)")
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Add text annotations with active feature counts
    for i, bar in enumerate(bars):
        feat_cnt = df["active_features_count"].iloc[i]
        mae = df["ridge_mae"].iloc[i]
        ax.text(mae + 0.02, bar.get_y() + bar.get_height()/2, f"{mae:.3f} ({feat_cnt} feats)", va='center', fontsize=8)

    plt.tight_layout()
    out_file = FIGURES_DIR / "feature_family_ablation_waterfall.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Generated: {out_file}")


def plot_modality_sensitivity():
    csv_path = TABLES_DIR / "modality_comparison_results.csv"
    if not csv_path.exists():
        print(f"Skipping plot_modality_sensitivity: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    modalities = ["corrected_transcript", "trocr_actual", "tesseract_actual"]
    mod_labels = ["Clean Transcript\n(Phase 8 Gold)", "TrOCR Hypothesis\n(Phase 9 Raw)", "Tesseract Hypothesis\n(Phase 9 Preprocessed)"]

    df = df.set_index("input_modality").loc[modalities].reset_index()

    x = np.arange(len(modalities))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # MAE comparison
    ax1.bar(x - width/2, df["ridge_mae"], width, label="Ridge Regression", color="#1f77b4", edgecolor="black")
    ax1.bar(x + width/2, df["rule_grader_mae"], width, label="Deterministic Rule Grader", color="#ff7f0e", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mod_labels)
    ax1.set_ylabel("Mean Absolute Error (Marks)")
    ax1.set_title("(A) MAE Across Input Representations")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(x)):
        ax1.text(x[i] - width/2, df["ridge_mae"].iloc[i] + 0.02, f"{df['ridge_mae'].iloc[i]:.2f}", ha='center', fontsize=9)
        ax1.text(x[i] + width/2, df["rule_grader_mae"].iloc[i] + 0.02, f"{df['rule_grader_mae'].iloc[i]:.2f}", ha='center', fontsize=9)

    # QWK comparison
    ax2.bar(x - width/2, df["ridge_qwk"], width, label="Ridge Regression", color="#2ca02c", edgecolor="black")
    ax2.bar(x + width/2, df["rule_grader_qwk"], width, label="Deterministic Rule Grader", color="#d62728", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mod_labels)
    ax2.set_ylabel("Quadratic Weighted Kappa (QWK)")
    ax2.set_title("(B) QWK Across Input Representations")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(x)):
        ax2.text(x[i] - width/2, df["ridge_qwk"].iloc[i] + 0.02, f"{df['ridge_qwk'].iloc[i]:.2f}", ha='center', fontsize=9)
        ax2.text(x[i] + width/2, df["rule_grader_qwk"].iloc[i] + 0.02, f"{df['rule_grader_qwk'].iloc[i]:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    out_file = FIGURES_DIR / "modality_sensitivity_slope.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Generated: {out_file}")


def plot_robustness_curves():
    csv_path = TABLES_DIR / "robustness_perturbation_results.csv"
    if not csv_path.exists():
        print(f"Skipping plot_robustness_curves: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    # Separate numeric noise from math sign flip
    df_noise = df[df["noise_condition"].str.startswith("CER_")].copy()
    df_noise["rate"] = [float(x.replace("CER_", "")) * 100 for x in df_noise["noise_condition"]]
    df_noise = df_noise.sort_values(by="rate")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = '#1f77b4'
    ax1.set_xlabel('Synthetic Character Substitution Rate (CER %)')
    ax1.set_ylabel('Ridge MAE (Marks)', color=color)
    line1 = ax1.plot(df_noise['rate'], df_noise['ridge_mae'], marker='o', color=color, linewidth=2, label="MAE")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = '#2ca02c'
    ax2.set_ylabel('Ridge QWK', color=color)
    line2 = ax2.plot(df_noise['rate'], df_noise['ridge_qwk'], marker='s', color=color, linewidth=2, linestyle='--', label="QWK")
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight math sign flip point if available
    flip_row = df[df["noise_condition"] == "math_sign_flip"]
    if not flip_row.empty:
        flip_mae = flip_row["ridge_mae"].iloc[0]
        ax1.axhline(flip_mae, color="red", linestyle=":", label=f"Math Sign Flip MAE ({flip_mae:.2f})")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')

    plt.title("Grading Robustness Under Progressive OCR Noise (Ridge Regressor)")
    plt.tight_layout()
    out_file = FIGURES_DIR / "robustness_noise_decay_curves.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Generated: {out_file}")


def plot_pareto_frontier():
    csv_path = TABLES_DIR / "pareto_efficiency_summary.csv"
    if not csv_path.exists():
        print(f"Skipping plot_pareto_frontier: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = df["latency_ms_per_eval"].values
    y = df["mae"].values
    sizes = np.clip(df["parameter_count_millions"].values * 0.1 + 40, 40, 500)

    # Color by Pareto status
    colors = ["#d62728" if p else "#1f77b4" for p in df["pareto_optimal_quality_vs_speed"]]

    scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.7, edgecolors="black")

    for i, row in df.iterrows():
        name = row["model_id"].replace("_", " ").title()
        ax.annotate(
            name, (row["latency_ms_per_eval"], row["mae"]),
            textcoords="offset points", xytext=(5, 5), fontsize=8
        )

    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency per Response (ms, log scale, lower is faster)")
    ax.set_ylabel("Mean Absolute Error (Marks, lower is better)")
    ax.set_title("Accuracy vs. Latency Pareto Frontier (Circle size = Param Count)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Pareto Dominant / Efficient', markerfacecolor='#d62728', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Standard Candidate', markerfacecolor='#1f77b4', markersize=10),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    out_file = FIGURES_DIR / "pareto_frontier_accuracy_latency.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Generated: {out_file}")


def generate_all_figures():
    print("Generating Paper 2 publication figures...")
    plot_overall_performance()
    plot_per_question_heatmap()
    plot_feature_ablation()
    plot_modality_sensitivity()
    plot_robustness_curves()
    plot_pareto_frontier()
    print("All figures successfully rendered.")


if __name__ == "__main__":
    generate_all_figures()
