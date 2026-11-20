"""
tests/test_phase1_manifests.py -- Verification tests for Phase 1 Manifests.
"""
import csv
import json
import pathlib
import hashlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "research_v2"
ANN = RES / "annotations"
CROPS = RES / "crops"


class TestPhase1Manifests(unittest.TestCase):
    def test_01_booklets_count_and_uniqueness(self):
        with open(ANN / "booklets.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 37, "Expected exactly 37 booklets")
        sids = [r["student_id"] for r in rows]
        self.assertEqual(len(sids), len(set(sids)), "Student IDs in booklets.csv must be unique")
        # Check split distribution
        train = [r for r in rows if r["split_or_fold_id"] == "train"]
        val = [r for r in rows if r["split_or_fold_id"] == "validation"]
        test = [r for r in rows if r["split_or_fold_id"] == "test"]
        self.assertEqual(len(train), 25)
        self.assertEqual(len(val), 6)
        self.assertEqual(len(test), 6)

    def test_02_pages_count_and_integrity(self):
        with open(ANN / "pages.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 195, "Expected exactly 195 pages")
        pids = [r["page_id"] for r in rows]
        self.assertEqual(len(pids), len(set(pids)), "Page IDs must be unique")
        for r in rows:
            self.assertGreater(int(r["width_px"]), 0)
            self.assertGreater(int(r["height_px"]), 0)
            self.assertEqual(len(r["image_sha256"]), 64)

    def test_03_marks_validity_and_sum(self):
        with open(ANN / "marks.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 148, "Expected 148 parent question mark records (37 x 4)")
        max_marks_map = {"Q1": 8.0, "Q2": 6.0, "Q3": 3.0, "Q4": 3.0}

        student_totals = {}
        for r in rows:
            sid = r["student_id"]
            qid = r["parent_question_id"]
            mark = float(r["teacher_mark"])
            max_m = float(r["maximum_mark"])
            self.assertEqual(max_m, max_marks_map[qid])
            self.assertGreaterEqual(mark, 0.0)
            self.assertLessEqual(mark, max_m)
            student_totals[sid] = student_totals.get(sid, 0.0) + mark

        # Verify booklet totals do not exceed 20.0
        for sid, total in student_totals.items():
            self.assertLessEqual(total, 20.0)

    def test_04_transcripts_integrity(self):
        with open(ANN / "transcripts.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 407, "Expected 407 transcript records (37 x 11)")
        for r in rows:
            self.assertIn(r["answer_status"], ["written", "omitted", "blank", "illegible", "unresolvable", "not_applicable"])
            if r["answer_status"] == "written":
                self.assertGreater(len(r["adjudicated_transcript"]), 0)

    def test_05_answer_regions_and_crop_hashes(self):
        with open(ANN / "answer_regions.csv", "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 407, "Expected 407 answer region records")

        # Check crops that exist
        verified_crops_tested = 0
        for r in rows:
            self.assertIn(r["annotation_status"], ["adjudicated", "unreviewed", "reviewed_by_a", "reviewed_by_b", "unresolvable"])
            c_rel = r["crop_relative_path"]
            if c_rel != "NONE":
                c_path = RES / c_rel
                self.assertTrue(c_path.exists(), f"Crop file missing: {c_path}")
                # Verify SHA-256 hash matches
                h = hashlib.sha256(c_path.read_bytes()).hexdigest()
                self.assertEqual(h, r["crop_sha256"])
                verified_crops_tested += 1

        self.assertGreater(verified_crops_tested, 300, "Expected most written crops to exist and verify")


if __name__ == "__main__":
    unittest.main()
