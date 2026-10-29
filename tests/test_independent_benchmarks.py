"""
tests/test_independent_benchmarks.py - Verification suite for Independent Multi-Modal Benchmarks.

Verifies:
1. 4-Stage Preprocessing execution & metrics (Red-ink separation, Deskew, Denoise, Sauvola).
2. CER / WER metric calculation precision.
3. Strict anti-leakage isolation (train and test student IDs strictly disjoint).
4. Integrity of all independent output JSON files and leaderboard consistency across all 27 models.
5. Flask endpoint /independent-benchmarks accessibility.
"""
import os
import sys
import json
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.independent_pipeline import (
    FourStageImagePreprocessor,
    compute_cer_wer,
    get_all_27_model_names
)
import app as flask_app_module


class TestIndependentBenchmarks(unittest.TestCase):

    def setUp(self):
        self.output_dir = os.path.join(BASE_DIR, "outputs", "independent_benchmarks")
        self.preprocessor = FourStageImagePreprocessor()

    def test_cer_wer_computation(self):
        """Verifies accuracy metrics return exact mathematical values."""
        # Exact match
        m1 = compute_cer_wer("hello world", "hello world")
        self.assertEqual(m1["cer"], 0.0)
        self.assertEqual(m1["wer"], 0.0)
        self.assertEqual(m1["char_acc"], 1.0)
        self.assertEqual(m1["word_acc"], 1.0)

        # Single substitution
        m2 = compute_cer_wer("hello warld", "hello world")
        self.assertAlmostEqual(m2["cer"], 1.0 / 11.0, places=3)
        self.assertEqual(m2["wer"], 0.5)

        # Completely empty
        m3 = compute_cer_wer("", "")
        self.assertEqual(m3["char_acc"], 1.0)

    def test_four_stage_preprocessing(self):
        """Verifies all 4 preprocessing stages produce valid arrays without crashing."""
        # Create synthetic BGR image with blue handwriting and red teacher mark
        synth = np.full((300, 300, 3), 255, dtype=np.uint8)
        # Blue line (student handwriting)
        synth[50:60, 50:250] = [180, 50, 20]
        # Red mark (teacher ink)
        synth[150:160, 50:250] = [20, 20, 220]

        res = self.preprocessor.process_image(synth)

        self.assertIn("stage1_student_clean_bgr", res)
        self.assertIn("stage1_teacher_marks_bgr", res)
        self.assertIn("stage2_deskewed_gray", res)
        self.assertIn("stage3_denoised_gray", res)
        self.assertIn("stage4_sauvola_bin", res)

        # Assert red ink was detected and isolated
        self.assertGreater(res["stage1_red_fraction"], 0.0)
        # Assert student clean has red ink removed
        self.assertEqual(res["stage1_student_clean_bgr"].shape, synth.shape)
        self.assertEqual(res["stage4_sauvola_bin"].shape, (300, 300))

    def test_strict_anti_leakage_manifest(self):
        """Verifies hard separation between training split and test split."""
        manifest_path = os.path.join(BASE_DIR, "outputs", "dataset_audit", "student_split_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), "student_split_manifest.json must exist")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        train_ids = set(manifest["splits"]["train"]["student_ids"])
        val_ids = set(manifest["splits"]["validation"]["student_ids"])
        test_ids = set(manifest["splits"]["test"]["student_ids"])

        # Hard assertion
        self.assertTrue(train_ids.isdisjoint(test_ids), "CRITICAL: train_ids and test_ids must be disjoint")
        self.assertTrue(val_ids.isdisjoint(test_ids), "CRITICAL: val_ids and test_ids must be disjoint")
        self.assertEqual(len(test_ids), 6, "Locked test split must have exactly 6 scripts")

    def test_independent_output_files_exist_and_valid(self):
        """Verifies that all required independent output JSON/CSV files exist and are populated."""
        expected_files = [
            "preprocessing_metrics.json",
            "cross_modality_comparison.json",
            "paddle_ocr_benchmark.json",
            "paddle_ocr_models_leaderboard.json",
            "paddle_ocr_models_leaderboard.csv",
            "paddle_ocr_predictions.json",
            "tesseract_ocr_benchmark.json",
            "tesseract_ocr_models_leaderboard.json",
            "tesseract_ocr_models_leaderboard.csv",
            "tesseract_ocr_predictions.json",
            "vlm_benchmark.json",
            "vlm_models_leaderboard.json",
            "vlm_models_leaderboard.csv",
            "vlm_predictions.json",
        ]

        for fname in expected_files:
            fp = os.path.join(self.output_dir, fname)
            self.assertTrue(os.path.exists(fp), f"Expected independent output file missing: {fname}")
            self.assertGreater(os.path.getsize(fp), 0, f"Independent output file is empty: {fname}")

    def test_all_27_models_present_in_leaderboard(self):
        """Verifies that all 27 models specified by user are evaluated and ranked in leaderboards."""
        paddle_lb_path = os.path.join(self.output_dir, "paddle_ocr_models_leaderboard.json")
        with open(paddle_lb_path, "r", encoding="utf-8") as f:
            lb = json.load(f)

        self.assertEqual(len(lb), 27, f"Leaderboard must have exactly 27 models, found {len(lb)}")
        model_names = {row["model_name"] for row in lb}
        expected_27 = {m[0] for m in get_all_27_model_names()}
        self.assertEqual(model_names, expected_27, "All 27 models must match the user specification")

    def test_flask_endpoint(self):
        """Verifies that /independent-benchmarks endpoint renders successfully with 200 OK."""
        client = flask_app_module.app.test_client()
        resp = client.get("/independent-benchmarks?format=json")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data.decode("utf-8"))
        self.assertIn("paddle_ocr", data)
        self.assertIn("tesseract_ocr", data)
        self.assertIn("vlm", data)
        self.assertEqual(len(data["paddle_leaderboard"]), 27)


if __name__ == "__main__":
    unittest.main()

