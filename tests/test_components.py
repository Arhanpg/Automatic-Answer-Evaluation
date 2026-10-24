import json, os, sys, tempfile, time, unittest
TMP = tempfile.mkdtemp(prefix="aae_test2_")
os.environ.update({"AAE_DATA_DIR": TMP, "AAE_MODE": "real"})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
import config
from backend import exam_parser, exam_schema as es, image_utils, database as db, pipeline, sampling, demo_data, ocr_engine, sheet_io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAM = exam_parser.build_exam_draft([os.path.join(ROOT, "examples/question_paper.pdf")],
                                    [os.path.join(ROOT, "examples/model_answer.pdf")])


class InkSplit(unittest.TestCase):
    def test_layers(self):
        img = Image.new("RGB", (800, 400), "white"); d = ImageDraw.Draw(img)
        d.rectangle([50, 50, 400, 70], fill=(20, 30, 160))        # student: blue ink
        d.rectangle([600, 100, 700, 130], fill=(200, 20, 30))     # teacher: red ink
        student, teacher, frac = image_utils.split_ink_layers(img, "red")
        s, t = np.asarray(student), np.asarray(teacher)
        self.assertTrue((s[110:120, 620:680] == 255).all())       # red gone from the answer layer
        self.assertTrue((s[55:65, 100:300] < 100).any())          # blue kept
        self.assertTrue((t[110:120, 620:680] < 50).all())         # red kept (as black) in the marks layer
        self.assertTrue((t[55:65, 100:300] == 255).all())         # blue absent from the marks layer
        self.assertGreater(frac, 0.005)

    def test_no_red(self):
        img = Image.new("RGB", (100, 100), "white")
        self.assertEqual(image_utils.split_ink_layers(img)[2], 0.0)


class ExamParse(unittest.TestCase):
    def test_sample_exam(self):
        self.assertEqual(es.answer_ids(EXAM), ["1a","1b","1c","1d","2a","2b","2c","3a","3b","3c","4"])
        self.assertEqual(EXAM["total_marks"], 20); self.assertEqual(es.total_marks(EXAM), 20)
        self.assertIn("No marks will be awarded", EXAM["rules"])
        self.assertIn("fixed, limited context window", EXAM["questions"][0]["parts"][0]["model_answer"])
        self.assertEqual(EXAM["warnings"], [])
        self.assertEqual(len(EXAM["read_warnings"]), 1)


class FolderOCR(unittest.TestCase):
    """Real folder layouts + the pipeline, with the OCR engine stubbed out."""
    def test_pipeline_with_stubbed_ocr(self):
        db.init_db(); es.save_exam(EXAM)
        src = tempfile.mkdtemp()
        from backend import pdf_utils
        Image.new("RGB", (900, 1200), "white").save(os.path.join(src, "s1.pdf"))        # one-page PDF per student
        os.makedirs(os.path.join(src, "s2")); [Image.new("RGB", (900, 1200), "white").save(os.path.join(src, "s2", f"{i}.jpg")) for i in (1, 2, 10)]
        Image.new("RGB", (900, 1200), "white").save(os.path.join(src, "21CS7_2.jpg")); Image.new("RGB", (900, 1200), "white").save(os.path.join(src, "21CS7_1.jpg"))
        units = sheet_io.collect_units(src)
        self.assertEqual([(u["roll"], len(u["files"])) for u in units], [("21CS7", 2), ("s1", 1), ("s2", 3)])
        self.assertEqual([os.path.basename(f) for f in units[2]["files"]], ["1.jpg", "2.jpg", "10.jpg"])   # natural page order

        def row(t, y, pg=0, x=40): return {"page": pg, "text": t, "x0": x, "x1": x + 300, "y0": y, "y1": y + 30, "yc": y + 15, "conf": 0.9}
        def fake_ocr(pages, split_ink=None, ink=None, progress=None):
            texts = ["1a) fixed context", "1b) word order", "1c) bidirectional", "1d) hides future", "2a) 1 0", "2b) 0.731 0.269",
                     "2c) 1.462 0.807", "3a) A", "3b) A B", "3c) normalisation", "4. P = 3/2420"]
            return {"answer_rows": [row(t, 100 + 80 * i) for i, t in enumerate(texts)],
                    "teacher_rows": [row("7/8", 360, x=700), row("5/6", 600, x=700), row("2/3", 840, x=700), row("3/3", 960, x=700)],
                    "teacher_fraction": [0.01], "mean_conf": 0.9}
        orig = ocr_engine.ocr_pages; ocr_engine.ocr_pages = fake_ocr
        try:
            pipeline.start_job("sheets", pipeline.job_process_sheets, src, True, "red")
            t0 = time.time()
            while time.time() - t0 < 20 and pipeline.get_progress()["state"] == "running": time.sleep(0.1)
        finally:
            ocr_engine.ocr_pages = orig
        p = pipeline.get_progress(); self.assertEqual(p["state"], "done", p)
        a = json.load(open(os.path.join(config.STUDENT_ANSWERS_DIR, "s1.json"), encoding="utf-8"))
        m = json.load(open(os.path.join(config.TEACHER_MARKS_DIR, "s1.json"), encoding="utf-8"))
        self.assertEqual(a["answers"]["1a"], "fixed context"); self.assertEqual(a["answers"]["4"], "P = 3/2420")
        self.assertEqual(m, {"roll_number": "s1", "marks": {"1": 7, "2": 5, "3": 2, "4": 3}})
        self.assertNotIn("marks", a); self.assertNotIn("answers", m)                     # folders stay separate
        self.assertEqual(db.counts()["total"], 3); self.assertEqual(sheet_io.count_saved_pages("s2"), 3)


class Selection(unittest.TestCase):
    def test_spread_and_label_free(self):
        studs = demo_data.make_students(EXAM, 40, seed=3)
        answers = {s["roll"]: s["answers"] for s in studs}
        sel, det, chart, used = sampling.select_training_papers(EXAM, answers, {}, 10)
        self.assertEqual(len(set(sel)), 10); self.assertIn(used, ("tfidf", "ollama-embeddings"))
        scores = {c["roll"]: c["score"] for c in chart}
        self.assertEqual(sel[0], min(scores, key=scores.get)); self.assertEqual(sel[-1], max(scores, key=scores.get))
        total = {s["roll"]: sum(v for v in s["marks"].values()) for s in studs}
        picked = sorted(total[r] for r in sel)
        self.assertLess(picked[0], 7); self.assertGreater(picked[-1], 14)             # covers weak AND strong students (by hidden marks)
        # the selector must not touch marks: identical result when marks are absent
        self.assertEqual(sel, sampling.select_training_papers(EXAM, answers, {}, 10)[0])


if __name__ == "__main__":
    unittest.main()
