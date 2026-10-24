"""End-to-end smoke test in DEMO mode with the real sample question paper + key (no OCR/GPU needed)."""
import json, os, sys, tempfile, time, unittest
TMP = tempfile.mkdtemp(prefix="aae_test_")
os.environ.update({"AAE_DATA_DIR": TMP, "AAE_MODE": "demo"})
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as appmod
from backend import pipeline, database as db


def wait(job, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = pipeline.get_progress()
        if p["job"] == job and p["state"] in ("done", "error"):
            return p
        time.sleep(0.1)
    raise TimeoutError(job)


class E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = appmod.app.test_client()

    def post(self, url, body=None, code=200):
        r = self.c.post(url, json=body or {})
        self.assertEqual(r.status_code, code, r.get_data(as_text=True)[:300])
        return r.get_json()

    def test_full_flow(self):
        for page in ("/", "/setup", "/sheets", "/select", "/training", "/results"):
            self.assertEqual(self.c.get(page).status_code, 200, page)
        # blocked until the exam exists
        self.post("/api/select", {"k": 5}, 400)
        # 1. exam
        self.post("/api/exam/use_sample"); self.post("/api/exam/parse")
        p = wait("exam"); self.assertEqual(p["state"], "done", p)
        draft = self.c.get("/api/exam/draft").get_json()["draft"]
        self.assertEqual([q["max_marks"] for q in draft["questions"]], [8, 6, 3, 3])
        saved = self.post("/api/exam/approve", {"exam": draft})["exam"]
        self.assertTrue(any("Q3(c)" in w for w in saved["read_warnings"])); self.assertEqual(saved["warnings"], [])
        # 2. sheets (demo)
        self.post("/api/sheets/demo", {"n": 30}); self.assertEqual(wait("sheets")["state"], "done")
        ans = json.load(open(os.path.join(TMP, "student_answers", "demo1.json"), encoding="utf-8"))
        mk = json.load(open(os.path.join(TMP, "teacher_marks", "demo1.json"), encoding="utf-8"))
        self.assertEqual(set(ans), {"roll_number", "answers"}); self.assertEqual(set(mk), {"roll_number", "marks"})
        self.assertEqual(list(ans["answers"]), ["1a","1b","1c","1d","2a","2b","2c","3a","3b","3c","4"])
        self.assertEqual(list(mk["marks"]), ["1", "2", "3", "4"])
        # 3. selection (label-free) + verify
        sel = self.post("/api/select", {"k": 8})
        self.assertEqual(len(sel["selected"]), 8)
        self.assertEqual(db.counts()["train"], 8); self.assertEqual(db.counts()["test"], 22)
        roll = sel["selected"][0]
        self.assertEqual(self.c.get(f"/verify/{roll}").status_code, 200)
        self.post(f"/api/verify/{roll}", {"answers": ans["answers"], "marks": {"1": 99}}, 400)   # > max rejected
        self.post(f"/api/verify/{roll}", {"answers": json.load(open(os.path.join(TMP, "student_answers", roll + ".json"), encoding="utf-8"))["answers"],
                                          "marks": json.load(open(os.path.join(TMP, "teacher_marks", roll + ".json"), encoding="utf-8"))["marks"]})
        self.assertEqual(db.counts()["train_verified"], 1)
        # 4. rubric + evaluation with the demo engine
        self.post("/api/rubrics/build"); self.assertEqual(wait("rubric")["state"], "done")
        self.post("/api/evaluate", {"engine": "hf:finetuned"}, 400)       # no adapter -> clear error, not a crash
        self.post("/api/evaluate", {"engine": "demo"}); self.assertEqual(wait("evaluate")["state"], "done")
        # 5. results: predictions exist ONLY for test papers; metrics computed from folder 2
        res = pipeline.compute_results("demo")
        self.assertEqual(res["metrics"]["n"], 22 * 4)
        tested = {r["roll"] for r in res["rows"]}
        self.assertTrue(tested.isdisjoint(set(sel["selected"])))
        self.assertLess(res["metrics"]["mae"], 3.0)
        # every page renders with real data in it
        for page, needle in (("/", "Folder 1"), ("/sheets", "demo1"), ("/select", "Training papers"),
                             ("/training", "Question 1"), ("/results", "Predicted vs actual"),
                             (f"/verify/{roll}", "Save"), (f"/results?run=demo", "Paper totals")):
            r = self.c.get(page); html = r.get_data(as_text=True)
            self.assertEqual(r.status_code, 200, page); self.assertIn(needle, html, page)
        self.assertEqual(self.c.get("/verify/nope").status_code, 404)
        self.assertEqual(self.c.get("/pages/demo1/1.jpg").status_code, 404)    # demo students have no scans, no crash

    def test_no_silent_fake_ocr(self):
        os.environ["AAE_MODE"] = "real"
        import importlib, config; importlib.reload(config)
        from backend import ocr_engine
        self.assertFalse(ocr_engine.paddleocr_available())
        with self.assertRaises(ocr_engine.OCRUnavailable):
            ocr_engine._build_engine()
        os.environ["AAE_MODE"] = "demo"; importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
