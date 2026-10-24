"""
run_full_evaluation.py - Master Orchestrator for Automated Answer Evaluation

Executes the complete end-to-end evaluation pipeline across all Milestones (M0 - M11):
- Phase 1 (M0, M1): Environment verification, exam schema validation, student-level split partitioning
- Phase 2 (M2): Four-Stage Hybrid Consensus OCR & Red-Ink extraction on IIIT Dharwad exam booklets
- Phase 3 (M3, M6, M4): Unified Semantic, SymPy Math Verifier, and Classical ML Benchmark (10 estimators + A0-A6 ablations)
- Phase 4 (M7, M8, M9): Evidence-constrained LLM Benchmark (Exp 71-82), Rubric RAG, and 4GB VRAM QLoRA adaptation
- Phase 5 (M10, M5, M11): Conformal Calibration, Human Review Routing, From-Scratch DL Lab, and Frontier Reproductions
- Integrity Verification: Asserts that all dedicated files in outputs/ and experiments/reports/ are populated and valid
"""

import os
import sys
import glob
import time
import argparse
import logging
from typing import Dict, Any, List, Tuple

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from backend.output_manager import DedicatedOutputManager
from backend.split_validation import initialize_and_save_splits, get_all_student_ids
from backend.dataset_validation import run_dataset_audit
from backend.ocr_engine import benchmark_ocr_engines
from backend.classical_ml import run_classical_ml_benchmark as run_m4_benchmark
from backend.grader import run_llm_grading_benchmark
from backend.rag_retriever import run_rag_ablation_experiments as run_rag_retrieval_experiment
from backend.qlora_trainer import run_qlora_data_ablation
from backend.uncertainty_calibration import generate_conformal_calibration
from backend.human_review_router import build_and_save_human_review_queue
from tests.test_gradients import run_m5_benchmarks_and_save
from src.frontier.preference_grading_dpo_grpo import run_grpo_preference_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("master_evaluator")

DEFAULT_DATASET_DIR = r"F:\Automatic answer evaluation dataset"


def verify_all_output_files(output_mgr: DedicatedOutputManager) -> Tuple[bool, List[str]]:
    """Verifies that all required files in outputs/ and experiments/ are populated and non-empty."""
    required_files = [
        # Dataset & Schema Audit
        os.path.join(output_mgr.dirs["dataset_audit"], "student_split_manifest.json"),
        os.path.join(output_mgr.dirs["dataset_audit"], "corpus_file_audit.json"),
        # OCR Metrics
        os.path.join(output_mgr.dirs["ocr_metrics"], "cer_wer_benchmark.json"),
        os.path.join(output_mgr.dirs["ocr_metrics"], "per_script_ocr_details.csv"),
        os.path.join(output_mgr.dirs["ocr_metrics"], "consensus_discrepancy_log.json"),
        os.path.join(output_mgr.dirs["ocr_metrics"], "red_ink_teacher_marks.json"),
        # Grading Performance
        os.path.join(output_mgr.dirs["grading_performance"], "model_leaderboard.csv"),
        os.path.join(output_mgr.dirs["grading_performance"], "question_wise_accuracy.json"),
        os.path.join(output_mgr.dirs["grading_performance"], "student_predictions_full.csv"),
        os.path.join(output_mgr.dirs["grading_performance"], "rubric_criterion_audit.json"),
        os.path.join(output_mgr.dirs["grading_performance"], "m4_feature_ablation_shap.json"),
        os.path.join(output_mgr.dirs["grading_performance"], "m7_llm_benchmark_summary.json"),
        os.path.join(output_mgr.dirs["grading_performance"], "m7_student_predictions.csv"),
        os.path.join(output_mgr.dirs["grading_performance"], "m8_rag_ablation_metrics.json"),
        os.path.join(output_mgr.dirs["grading_performance"], "m9_qlora_learning_curve.csv"),
        # Calibration & Reliability
        os.path.join(output_mgr.dirs["calibration_and_reliability"], "m10_conformal_intervals.json"),
        os.path.join(output_mgr.dirs["calibration_and_reliability"], "human_review_queue.json"),
        os.path.join(output_mgr.dirs["calibration_and_reliability"], "ocr_vs_verified_gap.json"),
        # From-Scratch DL Laboratory
        os.path.join(output_mgr.dirs["from_scratch_metrics"], "gradient_verification_log.json"),
        os.path.join(output_mgr.dirs["from_scratch_metrics"], "architecture_convergence.csv"),
        # Frontier Paper Reproductions
        os.path.join(output_mgr.dirs["frontier_reproductions"], "m11_quantization_pareto.csv"),
        os.path.join(output_mgr.dirs["frontier_reproductions"], "m11_grpo_vs_sft_dpo.json"),
        # System Telemetry
        os.path.join(output_mgr.dirs["system_telemetry"], "hardware_and_run_registry.json"),
        os.path.join(output_mgr.dirs["system_telemetry"], "m9_vram_profile.json"),
        # Markdown Reports
        os.path.join(output_mgr.dirs["reports"], "M1_dataset_and_rubric_audit.md"),
        os.path.join(output_mgr.dirs["reports"], "M2_ocr_report.md"),
        os.path.join(output_mgr.dirs["reports"], "M3_M6_unified_baseline.md"),
        os.path.join(output_mgr.dirs["reports"], "M4_classical_ml_report.md"),
        os.path.join(output_mgr.dirs["reports"], "M5_deep_learning_scratch.md"),
        os.path.join(output_mgr.dirs["reports"], "M7_llm_grading_report.md"),
        os.path.join(output_mgr.dirs["reports"], "M8_rag_retrieval_report.md"),
        os.path.join(output_mgr.dirs["reports"], "M9_finetuning_report.md"),
        os.path.join(output_mgr.dirs["reports"], "M10_reliability_report.md"),
        os.path.join(output_mgr.dirs["reports"], "M11_frontier_report.md"),
    ]

    missing_or_empty = []
    for fp in required_files:
        if not os.path.exists(fp):
            missing_or_empty.append(f"MISSING: {os.path.relpath(fp, BASE_DIR)}")
        elif os.path.getsize(fp) == 0:
            missing_or_empty.append(f"EMPTY: {os.path.relpath(fp, BASE_DIR)}")

    all_valid = (len(missing_or_empty) == 0)
    return all_valid, missing_or_empty


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Orchestrator (Milestones M0 - M11)")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR, help="Path to examination dataset")
    parser.add_argument("--max_ocr_scripts", type=int, default=3, help="Max scripts to process in live OCR")
    parser.add_argument("--skip_ocr", action="store_true", help="Skip live OCR if already completed")
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=================================================================")
    logger.info("STARTING MASTER EVALUATION ORCHESTRATOR - MILESTONES M0 TO M11")
    logger.info(f"Dataset Location: {args.dataset_dir}")
    logger.info(f"Project Root: {BASE_DIR}")
    logger.info("=================================================================")

    output_mgr = DedicatedOutputManager()

    # -------------------------------------------------------------
    # Milestone M1: Dataset Audit, Schema Validation & Student Splits
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 1: MILESTONE M1 - DATASET SCHEMA & STUDENT SPLITS <<<")
    splits = initialize_and_save_splits(output_mgr, args.dataset_dir)
    logger.info(f"Student splits initialized (SHA256: {splits['split_sha256'][:16]}...)")
    run_dataset_audit(output_mgr, args.dataset_dir)

    # -------------------------------------------------------------
    # Milestone M2: Four-Stage Hybrid Consensus OCR & Red-Ink Marks
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 2: MILESTONE M2 - FOUR-STAGE CONSENSUS OCR ENGINE <<<")
    if not args.skip_ocr:
        all_pdfs_folder = os.path.join(args.dataset_dir, "All PDF's")
        pdf_paths = glob.glob(os.path.join(all_pdfs_folder, "s*.pdf"))
        logger.info(f"Discovered {len(pdf_paths)} student examination PDFs in {all_pdfs_folder}")
        if pdf_paths:
            benchmark_ocr_engines(pdf_paths, output_mgr, max_scripts=args.max_ocr_scripts)
            logger.info("Milestone M2 OCR Benchmark completed and saved.")
    else:
        logger.info("Skipping OCR benchmark (using cached results).")

    # -------------------------------------------------------------
    # Milestone M3, M6 & M4: Semantic, SymPy Math & Classical ML
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 3: MILESTONES M3, M6 & M4 - UNIFIED GRADER & CLASSICAL ML <<<")
    m4_res = run_m4_benchmark(output_mgr)
    logger.info("Milestones M3/M6/M4 Classical ML Benchmark & A0-A6 ablations completed.")

    # -------------------------------------------------------------
    # Milestone M7 & M8: LLM Grading Benchmark & Rubric RAG
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 4: MILESTONES M7 & M8 - LLM BENCHMARK & RUBRIC RAG <<<")
    m7_res = run_llm_grading_benchmark(output_mgr)
    m8_res = run_rag_retrieval_experiment(output_mgr)
    logger.info("Milestones M7 (Exp 71-82) & M8 (Hybrid RAG) completed.")

    # -------------------------------------------------------------
    # Milestone M9: 4GB VRAM QLoRA Fine-Tuning & Data Volume Ablation
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 4 CONTINUED: MILESTONE M9 - 4GB QLORA ABLATION <<<")
    m9_res = run_qlora_data_ablation(output_mgr)
    logger.info("Milestone M9 QLoRA Data-Volume Ablation completed.")

    # -------------------------------------------------------------
    # Milestone M10: Conformal Prediction & Human Review Routing
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 5: MILESTONE M10 - CONFORMAL INTERVALS & HUMAN ROUTER <<<")
    m10_conf = generate_conformal_calibration(output_mgr)
    m10_queue = build_and_save_human_review_queue(output_mgr)
    logger.info("Milestone M10 Reliability & Human Review Gate completed.")

    # -------------------------------------------------------------
    # Milestone M5: Parallel Track B1 - From-Scratch DL Laboratory
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 5 CONTINUED: MILESTONE M5 - FROM-SCRATCH NEURAL LAB <<<")
    run_m5_benchmarks_and_save()
    logger.info("Milestone M5 Gradient Verification & Architecture Convergence completed.")

    # -------------------------------------------------------------
    # Milestone M11: Parallel Track B2 - Frontier Paper Reproductions
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 5 CONTINUED: MILESTONE M11 - FRONTIER REPRODUCTIONS <<<")
    m11_res = run_grpo_preference_experiment(output_mgr)
    logger.info("Milestone M11 Quantization Pareto, Mamba vs Transformer & GRPO completed.")

    # -------------------------------------------------------------
    # Verification Gate: Assert all 34 required dedicated files exist
    # -------------------------------------------------------------
    logger.info("\n=================================================================")
    logger.info("PERFORMING SYSTEM-WIDE INTEGRITY VERIFICATION GATE")
    logger.info("=================================================================")
    all_valid, missing_list = verify_all_output_files(output_mgr)

    total_time = round(time.time() - t_start, 2)
    if all_valid:
        logger.info(f"VERIFICATION SUCCESSFUL: All 34 required dedicated files exist, are populated and non-empty!")
        logger.info(f"Total Master Pipeline Runtime: {total_time}s")
    else:
        logger.error(f"VERIFICATION WARNING: {len(missing_list)} files missing or empty:")
        for item in missing_list:
            logger.error(f"  - {item}")

    logger.info("=================================================================")
    logger.info("AUTOMATED ANSWER EVALUATION PIPELINE RUN COMPLETED SUCCESSFULLY")
    logger.info("Flask Dashboard is ready at: http://localhost:5000/metrics-dashboard")
    logger.info("Outputs download available at: http://localhost:5000/download-outputs")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
