import json, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import exam_schema as es, sheet_parser as sp

EXAM = es.normalize_exam({"questions": [
    {"id": "1", "stem": "brief", "max_marks": 8, "parts": [{"id": c, "text": "t", "marks": 2} for c in "abcd"]},
    {"id": "2", "stem": "attn", "max_marks": 6, "parts": [{"id": c, "text": "t", "marks": 2} for c in "abc"]},
    {"id": "3", "stem": "lm", "max_marks": 3, "parts": [{"id": c, "text": "t", "marks": 1} for c in "abc"]},
    {"id": "4", "stem": "bigram", "max_marks": 3, "parts": []}]})


def rows(texts, page=0, x0=40, step=40, start=100, indent=None):
    out, y = [], start
    for t in texts:
        xi = indent.get(t, x0) if indent else x0
        out.append({"page": page, "text": t, "x0": xi, "x1": xi + 400, "y0": y, "y1": y + 30, "yc": y + 15, "conf": 0.9})
        y += step
    return out


def trow(text, y, page=0, x=500):
    return {"page": page, "text": text, "x0": x, "x1": x + 60, "y0": y, "y1": y + 30, "yc": y + 15, "conf": 0.9}


class Segmenting(unittest.TestCase):
    def test_clean_labels(self):
        r = rows(["Roll No: 23BCS045", "1a) fixed context", "1b) word order", "1c) bidirectional", "1d) hides future",
                  "2a) scores 1 0", "2b) softmax", "2c) 1.462 0.807", "3a) A", "3b) A B", "3c) normalises",
                  "4. P = 3/2420"])
        a, ev, fl = sp.segment_answers(EXAM, r)
        self.assertEqual(a["1a"], "fixed context"); self.assertEqual(a["3c"], "normalises")
        self.assertEqual(a["4"], "P = 3/2420"); self.assertFalse([f for f in fl if f.startswith("blank")])
        self.assertEqual(sp.find_roll_number(r), "23BCS045")

    def test_label_styles(self):
        r = rows(["Q1 (a) x1", "1.b. x2", "1) c) x3", "(d) x4", "Ans 2: a) y1", "b) y2", "c) y3", "3 a) z1", "3(b) z2", "3.c) z3",
                  "Q.4) w"])
        a, _, fl = sp.segment_answers(EXAM, r)
        self.assertEqual([a[k] for k in ("1a", "1b", "1c", "1d")], ["x1", "x2", "x3", "x4"])
        self.assertEqual([a[k] for k in ("2a", "2b", "2c")], ["y1", "y2", "y3"])
        self.assertEqual([a[k] for k in ("3a", "3b", "3c", "4")], ["z1", "z2", "z3", "w"])

    def test_continuation_and_false_positives(self):
        r = rows(["1a) fixed", "context window", "1b) order", "2a) Q.K1 = 1", "1) not a label (Q1 already used)",
                  "0.731 and 0.269", "2b) softmax", "2c) out"], indent={"1) not a label (Q1 already used)": 120})
        a, _, _ = sp.segment_answers(EXAM, r)
        self.assertEqual(a["1a"], "fixed\ncontext window")
        self.assertIn("1) not a label", a["2a"]); self.assertIn("0.731", a["2a"])

    def test_missed_question_number_recovers(self):
        r = rows(["1a) a", "1b) b", "1c) c", "1d) d", "a) scores", "b) softmax", "c) final"])
        a, _, _ = sp.segment_answers(EXAM, r)
        self.assertEqual((a["2a"], a["2c"]), ("scores", "final"))

    def test_out_of_order_and_blank(self):
        r = rows(["4. bigram", "3a) A", "3c) norm"])
        a, _, fl = sp.segment_answers(EXAM, r)
        self.assertEqual((a["4"], a["3a"], a["3b"], a["3c"]), ("bigram", "A", "", "norm"))
        self.assertTrue(any(f.startswith("blank_parts") and "3b" in f for f in fl))

    def test_no_labels(self):
        _, _, fl = sp.segment_answers(EXAM, rows(["just some text", "more"]))
        self.assertTrue(any(f.startswith("no_labels_found") for f in fl))


class Marks(unittest.TestCase):
    def setUp(self):
        r = rows(["1a) a", "1b) b", "1c) c", "1d) d", "2a) a", "2b) b", "2c) c", "3a) a", "3b) b", "3c) c", "4. w"],
                 step=100, start=100)
        self.rows = r
        _, self.ev, _ = sp.segment_answers(EXAM, r)

    def test_fractions(self):
        t = [trow("7/8", 440), trow("5/6", 740), trow("2 / 3", 1040), trow("3/3", 1140), trow("Total 17/20", 20)]
        m, fl = sp.extract_marks(EXAM, t, self.ev)
        self.assertEqual(m, {"1": 7, "2": 5, "3": 2, "4": 3}); self.assertEqual(fl, [])

    def test_bare_numbers_next_to_labels(self):
        t = [trow("7", 100), trow("5", 500), trow("2", 800), trow("3", 1100)]
        m, _ = sp.extract_marks(EXAM, t, self.ev)
        self.assertEqual(m, {"1": 7, "2": 5, "3": 2, "4": 3})

    def test_part_marks_are_summed(self):
        t = [trow("2", 120), trow("1.5", 220), trow("2", 320), trow("1", 420)]
        m, _ = sp.extract_marks(EXAM, t, self.ev)
        self.assertEqual(m["1"], 6.5)

    def test_flags(self):
        t = [trow("7/8", 440), trow("Total 19", 20)]
        m, fl = sp.extract_marks(EXAM, t, self.ev)
        self.assertIsNone(m["3"]); self.assertTrue(any("marks_Q3" in f for f in fl))
        m, fl = sp.extract_marks(EXAM, [], self.ev)
        self.assertTrue(fl and all(v is None for v in m.values()))

    def test_cover_table(self):
        t = [trow("Q 1 2 3 4 Total", 10), trow("7 5 2 3 17", 40)]
        m, fl = sp.extract_marks(EXAM, t, self.ev)
        self.assertEqual(m, {"1": 7, "2": 5, "3": 2, "4": 3})


if __name__ == "__main__":
    unittest.main()
