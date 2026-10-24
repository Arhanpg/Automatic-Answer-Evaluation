"""
backend/output_manager.py -- Centralized Dedicated Output Manager.

Ensures every pipeline execution, OCR pass, baseline model, and research experiment
atomically writes structured JSON, CSV, and Markdown telemetry to dedicated files
across outputs/ and experiments/.
"""
import os
import json
import csv
import time
import hashlib
from typing import Any, Dict, List, Optional
import pandas as pd

class DedicatedOutputManager:
    """Manages the standardized output folder hierarchy and atomic serialization."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # automatic-answer-evaluation/automatic-answer-evaluation root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = os.path.abspath(base_dir)

        # Standard subdirectories mandated across Milestones M0 - M11
        self.dirs = {
            "dataset_audit": os.path.join(self.base_dir, "outputs", "dataset_audit"),
            "ocr_metrics": os.path.join(self.base_dir, "outputs", "ocr_metrics"),
            "grading_performance": os.path.join(self.base_dir, "outputs", "grading_performance"),
            "calibration_and_reliability": os.path.join(self.base_dir, "outputs", "calibration_and_reliability"),
            "from_scratch_metrics": os.path.join(self.base_dir, "outputs", "from_scratch_metrics"),
            "frontier_reproductions": os.path.join(self.base_dir, "outputs", "frontier_reproductions"),
            "system_telemetry": os.path.join(self.base_dir, "outputs", "system_telemetry"),
            "reports": os.path.join(self.base_dir, "experiments", "reports"),
            "runs": os.path.join(self.base_dir, "experiments", "runs"),
            "configs": os.path.join(self.base_dir, "experiments", "configs"),
        }

        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

    @property
    def dataset_dir(self) -> str:
        return self.dirs["dataset_audit"]

    @property
    def ocr_dir(self) -> str:
        return self.dirs["ocr_metrics"]

    @property
    def grading_dir(self) -> str:
        return self.dirs["grading_performance"]

    @property
    def calibration_dir(self) -> str:
        return self.dirs["calibration_and_reliability"]

    @property
    def system_dir(self) -> str:
        return self.dirs["system_telemetry"]

    @property
    def reports_dir(self) -> str:
        return self.dirs["reports"]

    def _write_json(self, file_path: str, data: Any):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        temp_path = file_path + f".tmp.{os.getpid()}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)

    def _write_csv(self, file_path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not rows:
            if fieldnames:
                df = pd.DataFrame(columns=fieldnames)
                df.to_csv(file_path, index=False)
            return

        df = pd.DataFrame(rows)
        if fieldnames:
            cols = [c for c in fieldnames if c in df.columns] + [c for c in df.columns if c not in fieldnames]
            df = df[cols]
        temp_path = file_path + f".tmp.{os.getpid()}"
        df.to_csv(temp_path, index=False, quoting=csv.QUOTE_MINIMAL)
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)

    def save_ocr_benchmark(
        self,
        metrics_dict: Dict[str, Any],
        per_script_rows: List[Dict[str, Any]],
        discrepancy_rows: List[Dict[str, Any]],
    ):
        """Writes outputs/ocr_metrics/cer_wer_benchmark.json, per_script_ocr_details.csv, and consensus_discrepancy_log.json."""
        self._write_json(os.path.join(self.dirs["ocr_metrics"], "cer_wer_benchmark.json"), metrics_dict)
        self._write_csv(os.path.join(self.dirs["ocr_metrics"], "per_script_ocr_details.csv"), per_script_rows)
        self._write_json(os.path.join(self.dirs["ocr_metrics"], "consensus_discrepancy_log.json"), discrepancy_rows)

    def save_grading_performance(
        self,
        leaderboard_rows: List[Dict[str, Any]],
        question_breakdown: Dict[str, Any],
        student_predictions: List[Dict[str, Any]],
        criterion_audit: List[Dict[str, Any]],
    ):
        """Writes outputs/grading_performance/model_leaderboard.csv, question_wise_accuracy.json,
        student_predictions_full.csv, and rubric_criterion_audit.json."""
        self._write_csv(os.path.join(self.dirs["grading_performance"], "model_leaderboard.csv"), leaderboard_rows)
        self._write_json(os.path.join(self.dirs["grading_performance"], "question_wise_accuracy.json"), question_breakdown)
        self._write_csv(os.path.join(self.dirs["grading_performance"], "student_predictions_full.csv"), student_predictions)
        self._write_json(os.path.join(self.dirs["grading_performance"], "rubric_criterion_audit.json"), criterion_audit)

    def save_reliability_metrics(
        self,
        ocr_gap_dict: Dict[str, Any],
        review_queue_list: List[Dict[str, Any]],
        conformal_dict: Dict[str, Any],
    ):
        """Writes outputs/calibration_and_reliability/ocr_vs_verified_gap.json,
        human_review_queue.json, and m10_conformal_intervals.json."""
        self._write_json(os.path.join(self.dirs["calibration_and_reliability"], "ocr_vs_verified_gap.json"), ocr_gap_dict)
        self._write_json(os.path.join(self.dirs["calibration_and_reliability"], "human_review_queue.json"), review_queue_list)
        self._write_json(os.path.join(self.dirs["calibration_and_reliability"], "m10_conformal_intervals.json"), conformal_dict)

    def save_dataset_audit(self, manifest_dict: Dict[str, Any]):
        """Writes outputs/dataset_audit/student_split_manifest.json."""
        self._write_json(os.path.join(self.dirs["dataset_audit"], "student_split_manifest.json"), manifest_dict)

    def save_red_ink_marks(self, marks_dict: Dict[str, Any]):
        """Writes outputs/ocr_metrics/red_ink_teacher_marks.json."""
        self._write_json(os.path.join(self.dirs["ocr_metrics"], "red_ink_teacher_marks.json"), marks_dict)

    def save_from_scratch_metrics(self, grad_log: Dict[str, Any], convergence_rows: List[Dict[str, Any]]):
        """Writes outputs/from_scratch_metrics/gradient_verification_log.json and architecture_convergence.csv."""
        self._write_json(os.path.join(self.dirs["from_scratch_metrics"], "gradient_verification_log.json"), grad_log)
        self._write_csv(os.path.join(self.dirs["from_scratch_metrics"], "architecture_convergence.csv"), convergence_rows)

    def save_frontier_reproductions(self, quant_pareto_rows: List[Dict[str, Any]], grpo_dict: Dict[str, Any]):
        """Writes outputs/frontier_reproductions/m11_quantization_pareto.csv and m11_grpo_vs_sft_dpo.json."""
        self._write_csv(os.path.join(self.dirs["frontier_reproductions"], "m11_quantization_pareto.csv"), quant_pareto_rows)
        self._write_json(os.path.join(self.dirs["frontier_reproductions"], "m11_grpo_vs_sft_dpo.json"), grpo_dict)

    def save_qlora_metrics(self, curve_rows: List[Dict[str, Any]], vram_profile: Dict[str, Any]):
        """Writes outputs/grading_performance/m9_qlora_learning_curve.csv and outputs/system_telemetry/m9_vram_profile.json."""
        self._write_csv(os.path.join(self.dirs["grading_performance"], "m9_qlora_learning_curve.csv"), curve_rows)
        self._write_json(os.path.join(self.dirs["system_telemetry"], "m9_vram_profile.json"), vram_profile)

    def save_report(self, report_filename: str, markdown_content: str):
        """Writes a markdown report to experiments/reports/{report_filename}."""
        target = os.path.join(self.dirs["reports"], report_filename)
        with open(target, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def register_experiment_run(self, run_metadata: Dict[str, Any], markdown_summary: Optional[str] = None):
        """Appends to outputs/system_telemetry/hardware_and_run_registry.json,
        writes experiments/runs/{milestone}/{experiment_id}.json, and updates report."""
        registry_path = os.path.join(self.dirs["system_telemetry"], "hardware_and_run_registry.json")
        entries = []
        if os.path.exists(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    if not isinstance(entries, list):
                        entries = [entries]
            except Exception:
                entries = []

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **run_metadata,
        }
        entries.append(entry)
        self._write_json(registry_path, entries)

        milestone = run_metadata.get("milestone", "misc")
        exp_id = run_metadata.get("experiment_id", f"run_{int(time.time())}")
        milestone_run_dir = os.path.join(self.dirs["runs"], milestone)
        os.makedirs(milestone_run_dir, exist_ok=True)
        self._write_json(os.path.join(milestone_run_dir, f"{exp_id}.json"), run_metadata)

        if markdown_summary:
            rep_path = os.path.join(self.dirs["reports"], f"{milestone}_report.md")
            with open(rep_path, "w", encoding="utf-8") as f:
                f.write(markdown_summary)
