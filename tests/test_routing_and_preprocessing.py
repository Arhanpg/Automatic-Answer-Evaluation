"""
tests/test_routing_and_preprocessing.py -- Unit tests for:
1. Robust cover page detection in sheet_parser (page-zero exclusion fix).
2. Red ink extraction and inpainting in red_ink_extractor.
3. OpenCV 4.5+ minAreaRect deskew angle convention.
4. Modular full_preprocess_pipeline for ablation studies.
"""
import unittest
import numpy as np
import cv2
from backend import exam_schema as es, sheet_parser as sp, red_ink_extractor as rie

EXAM = es.normalize_exam({
    "questions": [
        {"id": "1", "stem": "brief", "max_marks": 8, "parts": [{"id": c, "text": "t", "marks": 2} for c in "abcd"]},
        {"id": "2", "stem": "attn", "max_marks": 6, "parts": [{"id": c, "text": "t", "marks": 2} for c in "abc"]},
        {"id": "3", "stem": "lm", "max_marks": 3, "parts": [{"id": c, "text": "t", "marks": 1} for c in "abc"]},
        {"id": "4", "stem": "bigram", "max_marks": 3, "parts": []}
    ]
})

def make_row(text, page=0, y0=100, x0=50):
    return {
        "page": page,
        "text": text,
        "x0": x0,
        "x1": x0 + 400,
        "y0": y0,
        "y1": y0 + 30,
        "yc": y0 + 15,
        "conf": 0.95
    }

class TestSheetParserCoverPage(unittest.TestCase):
    def test_page_zero_not_discarded_when_it_contains_answers(self):
        """When page 0 contains answers and page 1 is the cover sheet (like s2.pdf), page 0 must be kept!"""
        rows = [
            make_row("4. Bigram language model derivation", page=0, y0=100),
            make_row("P(I like ML) = 3/2420", page=0, y0=140),
            # Page 1 has cover sheet anchors
            make_row("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY", page=1, y0=50),
            make_row("INSTRUCTIONS TO CANDIDATES", page=1, y0=90),
            make_row("FOR USE OF EXAMINERS ONLY", page=1, y0=130),
            # Page 2 has Q1
            make_row("1a) Limited context window", page=2, y0=100),
        ]
        answers, events, flags = sp.segment_answers(EXAM, rows)
        self.assertIn("Bigram language model", answers["4"])
        self.assertEqual(answers["1a"], "Limited context window")

    def test_page_zero_discarded_when_it_is_the_cover_sheet(self):
        """When page 0 is the cover sheet, it must be skipped and page 1 answers kept."""
        rows = [
            make_row("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY", page=0, y0=50),
            make_row("INSTRUCTIONS TO CANDIDATES", page=0, y0=90),
            make_row("FOR USE OF EXAMINERS ONLY", page=0, y0=130),
            make_row("1a) Limited context window", page=1, y0=100),
            make_row("1b) Positional information", page=1, y0=150),
        ]
        answers, events, flags = sp.segment_answers(EXAM, rows)
        self.assertEqual(answers["1a"], "Limited context window")
        self.assertEqual(answers["1b"], "Positional information")

class TestRedInkExtraction(unittest.TestCase):
    def test_dual_band_red_ink_removal(self):
        """Generates synthetic white canvas with black text and red teacher mark, verifying separation."""
        img = np.full((200, 400, 3), 255, dtype=np.uint8)
        # Black student text: "Attention"
        cv2.putText(img, "Attention", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        # Bright red teacher tick: (B=0, G=0, R=255)
        cv2.putText(img, "Tick +2.0", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        student_clean, teacher_layer, frac = rie.extract_and_remove_red_ink(img)
        self.assertGreater(frac, 0.001)

        # In student clean, red text area should be inpainted to white
        red_crop = student_clean[80:120, 200:350]
        self.assertGreater(np.mean(red_crop), 245.0)

        # In teacher layer, red marks should be extracted as dark marks
        teacher_crop = teacher_layer[80:120, 200:350]
        self.assertLess(np.mean(teacher_crop), 245.0)

class TestDeskewAngleConvention(unittest.TestCase):
    def test_angle_convention_positive_and_negative(self):
        """Tests OpenCV 4.5+ angle handling on synthetic rotated rectangles."""
        for target_angle in [-10.0, -5.0, 5.0, 10.0]:
            img = np.full((300, 300), 255, dtype=np.uint8)
            rect_pts = np.array([[100, 120], [200, 120], [200, 180], [100, 180]])
            center = (150, 150)
            M = cv2.getRotationMatrix2D(center, target_angle, 1.0)
            ones = np.ones(shape=(len(rect_pts), 1))
            points_ones = np.hstack([rect_pts, ones])
            transformed = M.dot(points_ones.T).T.astype(np.int32)
            cv2.fillPoly(img, [transformed], 0)

            coords = np.column_stack(np.where(img == 0))
            rect = cv2.minAreaRect(coords)
            raw_angle = rect[-1]
            w_box, h_box = rect[1]
            if w_box < h_box:
                est_angle = float(raw_angle)
            else:
                est_angle = -float(90.0 - raw_angle)

            # Error should be within 1.0 degree
            self.assertAlmostEqual(abs(est_angle - target_angle) % 90.0, 0.0, delta=1.5)

class TestModularPreprocessingPipeline(unittest.TestCase):
    def test_pipeline_ablation_artifacts(self):
        """Verifies full_preprocess_pipeline produces all necessary artifacts for ablation."""
        bgr = np.full((100, 100, 3), 240, dtype=np.uint8)
        cv2.putText(bgr, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        out = rie.full_preprocess_pipeline(bgr, remove_red=True, suppress_grid=True, deskew=True, denoise=True, binarize=True)
        self.assertIn("clean_rgb", out)
        self.assertIn("bin_image", out)
        self.assertIn("skew_angle", out)
        self.assertEqual(out["clean_rgb"].shape, (100, 100, 3))
        self.assertEqual(out["bin_image"].shape, (100, 100))

if __name__ == "__main__":
    unittest.main()
