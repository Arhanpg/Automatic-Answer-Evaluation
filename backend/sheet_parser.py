"""
backend/sheet_parser.py -- OCR rows of ONE student's script  ->  folder-1 + folder-2 data.

Why this works better than the old regex pass: the parser is GUIDED BY THE EXAM.
It only accepts labels that exist in the question paper (1a, 1b ... 4), never
re-opens a question it already closed, understands many handwriting styles
("1a)", "Q1 (a)", "1.a.", "Ans 4:", a bare "(b)") and recovers when a question
number was missed.

Input rows (from ocr_engine.merge_rows): {"page","text","x0","y0","x1","y1","yc","conf"}
  answer_rows  - student's writing, teacher's ink removed
  teacher_rows - only the teacher's ink (marks)

Output:
  answers : {"1a": "...", "1b": "...", ..., "4": "..."}      -> data/student_answers/<id>.json
  marks   : {"1": 7, "2": 5, "3": null, "4": 3}               -> data/teacher_marks/<id>.json
  flags   : QC warnings kept in the DB (NOT in the two clean folders)
"""
import re

from backend import exam_schema as es

_PREFIX = r"(?:(?:q(?:uestion|ns?)?|ans(?:wer)?)\s*[\.\-:]?\s*)?"
FULL_RE = re.compile(rf"^\s*{_PREFIX}(\d{{1,2}})\s*[\.\)\-:]?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]\.:]\s*(.*)$", re.I)
Q_RE = re.compile(rf"^\s*{_PREFIX}(\d{{1,2}})\s*[\.\)\:\-](?!\d)\s*(.*)$", re.I)
QWORD_RE = re.compile(r"^\s*(?:q(?:uestion|ns?)?|ans(?:wer)?)\s*[\.\-:]?\s*(\d{1,2})\b\s*[\.\)\:\-]?\s*(.*)$", re.I)
SUB_RE = re.compile(r"^\s*[\(\[]?\s*([A-Za-z])\s*[\)\]\.:]\s*(.*)$")
ROLL_RE = re.compile(r"(?:roll|reg(?:istration)?|enrol+ment|usn)\s*(?:no|number|num)?\.?\s*[:\-=]?\s*"
                     r"([A-Za-z0-9][A-Za-z0-9\-/]{3,14})", re.I)
FRAC_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*[/\\|]\s*(\d{1,2})(?!\d)")
NUM_RE = re.compile(r"(?<![\d./])(\d{1,2}(?:\.\d)?)(?![\d/])")


def _clean(s):
    return re.sub(r"\s+", " ", s.replace("\u200b", "")).strip()


# ---------------------------------------------------------------------------
# roll number
# ---------------------------------------------------------------------------
def find_roll_number(rows):
    for r in rows[:30]:
        if r["page"] != rows[0]["page"]:
            break
        m = ROLL_RE.search(r["text"])
        if m and not m.group(1).lower() in {"number", "no"}:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# answers
# ---------------------------------------------------------------------------
class _Segmenter:
    def __init__(self, exam, rows):
        self.exam = exam
        self.q_parts = {q["id"]: [p["id"] for p in q["parts"]] for q in exam["questions"]}
        self.rows = rows
        self.page_stats = {}
        for pg in {r["page"] for r in rows}:
            xs = sorted(r["x0"] for r in rows if r["page"] == pg)
            x1 = max(r["x1"] for r in rows if r["page"] == pg)
            left = xs[int(0.15 * (len(xs) - 1))]
            self.page_stats[pg] = (left, max(1.0, x1 - xs[0]))
        self.seen_q, self.seen_ids = set(), set()
        self.cur_q = None
        self.cur_id = None
        self.buf, self.events = {}, []

    def _flush_left(self, row):
        left, width = self.page_stats.get(row["page"], (0, 1))
        return row["x0"] <= left + 0.35 * width

    def _match(self, text, row):
        clean_text = re.sub(r"^[^\w\(\[]+\s*", "", text)
        if clean_text:
            text = clean_text

        flush = self._flush_left(row)
        m = FULL_RE.match(text)
        if m and flush:
            q, sub = m.group(1), m.group(2).lower()
            if sub in self.q_parts.get(q, []) and (q + sub) not in self.seen_ids:
                return q, sub, m.group(3)
        m = Q_RE.match(text) or QWORD_RE.match(text)
        if m and flush:
            q, rest = m.group(1), m.group(2)
            if q in self.q_parts and q not in self.seen_q:
                ms = SUB_RE.match(rest)
                if ms and ms.group(1).lower() in self.q_parts[q]:
                    return q, ms.group(1).lower(), ms.group(2)
                return q, None, rest
        m = SUB_RE.match(text)
        if m:
            sub, rest = m.group(1).lower(), m.group(2)
            if not self.cur_q:
                first_q = next((qid for qid, parts in self.q_parts.items() if sub in parts), None)
                if first_q:
                    return first_q, sub, rest
            elif sub in self.q_parts[self.cur_q] and (self.cur_q + sub) not in self.seen_ids:
                return self.cur_q, sub, rest
            elif sub == "a":   # a new "(a)" while we're already past (a): the "2." was missed
                for qid, parts in self.q_parts.items():
                    if qid not in self.seen_q and "a" in parts and self._after(qid):
                        return qid, "a", rest
        return None

    def _after(self, qid):
        order = list(self.q_parts)
        return order.index(qid) > order.index(self.cur_q)

    def _open(self, q, sub, row):
        aid = q + sub if sub else q
        self.cur_q, self.cur_id = q, aid
        self.seen_q.add(q)
        self.seen_ids.add(aid)
        self.buf.setdefault(aid, [])
        self.events.append({"page": row["page"], "y": row["y0"], "yc": row["yc"], "q": q, "id": aid})

    def run(self):
        has_later_pages = any(r["page"] > 0 for r in self.rows)
        for row in self.rows:
            if has_later_pages and row["page"] == 0:
                continue
            text = _clean(row["text"])
            if not text:
                continue
            label = self._match(text, row)
            if label:
                q, sub, rest = label
                self._open(q, sub, row)
                if rest.strip():
                    self.buf[self.cur_id].append(rest.strip())
            elif self.cur_id is not None:
                self.buf[self.cur_id].append(text)
        return self.buf, self.events


def segment_answers(exam, rows):
    """-> (answers {id: text}, events, flags)"""
    buf, events = _Segmenter(exam, rows).run()
    flags = []
    answers = {aid: "" for aid in es.answer_ids(exam)}
    for q in exam["questions"]:
        qid = q["id"]
        if q["parts"]:
            pseudo = "\n".join(buf.get(qid, []))
            for p in q["parts"]:
                answers[qid + p["id"]] = "\n".join(buf.get(qid + p["id"], [])).strip()
            if pseudo and not any(answers[qid + p["id"]] for p in q["parts"]):
                first = qid + q["parts"][0]["id"]
                answers[first] = pseudo.strip()
                flags.append(f"Q{qid}: no sub-question letters found - text placed under {first}")
        else:
            answers[qid] = "\n".join(buf.get(qid, [])).strip()
    if not events:
        flags.append("no_labels_found: could not find any question numbers")
    blank = [aid for aid, t in answers.items() if not t]
    if blank and events:
        flags.append("blank_parts: " + ", ".join(blank))
    return answers, events, flags


# ---------------------------------------------------------------------------
# teacher marks
# ---------------------------------------------------------------------------
def _tokens(row):
    text = row["text"]
    fracs = [(float(m.group(1)), float(m.group(2))) for m in FRAC_RE.finditer(text)]
    rest = FRAC_RE.sub(" ", text)
    nums = [float(m.group(1)) for m in NUM_RE.finditer(rest)]
    return fracs, nums


def _region_of(row, events):
    """Which question does this teacher-ink row belong to? (last label above/at it)"""
    owner = None
    slack = 0.25 * (row["y1"] - row["y0"])      # a mark may sit a hair above its label's baseline
    for e in events:
        if (e["page"], e["y"] - slack) <= (row["page"], row["yc"]):
            owner = e["q"]
        else:
            break
    return owner


def _choose(exam_q, fracs, nums):
    mx = exam_q["max_marks"]
    if mx is None:
        return None, "no max marks in exam"
    exact = [n for n, d in fracs if d == mx and n <= mx]
    if exact:
        return exact[-1], None
    if fracs and abs(sum(d for _, d in fracs) - mx) < 1e-9:
        return sum(n for n, _ in fracs), None
    nums = [n for n in nums if n <= mx]
    if len(nums) == 1:
        return nums[0], None
    if len(nums) > 1:
        pm = [p["marks"] for p in exam_q["parts"]]
        if len(nums) == len(pm) and all(pm) and all(n <= m for n, m in zip(nums, pm)) and sum(nums) <= mx:
            return sum(nums), "summed part marks"
        return nums[-1], "ambiguous (several numbers)"
    return None, "missing"


def _marks_table(exam, rows):
    """Cover-page table: a row of numbers [m1 m2 ... mN (total)]."""
    qs = exam["questions"]
    for r in rows:
        fracs, nums = _tokens(r)
        if len(nums) in (len(qs), len(qs) + 1) and not fracs:
            vals = nums[:len(qs)]
            if all(q["max_marks"] is not None and v <= q["max_marks"] for q, v in zip(qs, vals)):
                if len(nums) == len(qs) + 1 and abs(sum(vals) - nums[-1]) > 1e-9:
                    continue
                return {q["id"]: v for q, v in zip(qs, vals)}
    return None


def extract_marks(exam, teacher_rows, events):
    """-> (marks {qid: number|None}, flags)"""
    flags = []
    marks = {q["id"]: None for q in exam["questions"]}
    if not teacher_rows:
        return marks, ["no_teacher_marks_found: no red ink detected - enter marks manually or import a CSV"]

    rows = sorted(teacher_rows, key=lambda r: (r["page"], r["yc"], r["x0"]))
    per_q = {q["id"]: {"fracs": [], "nums": []} for q in exam["questions"]}
    preamble, total_hint = [], None
    for r in rows:
        fracs, nums = _tokens(r)
        if re.search(r"total|obtained|grand", r["text"], re.I):
            vals = [n for n, _ in fracs] + nums
            if vals:
                total_hint = vals[-1]
            continue
        owner = _region_of(r, events)
        if owner is None:
            preamble.append(r)
        else:
            per_q[owner]["fracs"] += fracs
            per_q[owner]["nums"] += nums

    for q in exam["questions"]:
        val, note = _choose(q, per_q[q["id"]]["fracs"], per_q[q["id"]]["nums"])
        marks[q["id"]] = val
        if note and note != "summed part marks":
            flags.append(f"marks_Q{q['id']}: {note}")

    if all(v is None for v in marks.values()):
        table = _marks_table(exam, preamble)
        if table:
            marks = table
            flags = [f for f in flags if not f.startswith("marks_Q")] + ["marks read from cover-page table"]

    got = [v for v in marks.values() if v is not None]
    if total_hint is not None and got and len(got) == len(marks) and abs(sum(got) - total_hint) > 1e-9:
        flags.append(f"marks_sum_mismatch: questions add up to {sum(got):g} but the script total reads {total_hint:g}")
    return marks, flags


# ---------------------------------------------------------------------------
def parse_student(exam, ocr):
    """ocr = output of ocr_engine.ocr_pages()  ->  dict(answers, marks, flags, roll_hint)"""
    rows = ocr["answer_rows"]
    answers, events, flags = segment_answers(exam, rows)
    marks, mflags = extract_marks(exam, ocr["teacher_rows"], events)
    flags += mflags
    if ocr["mean_conf"] and ocr["mean_conf"] < 0.6:
        flags.append(f"low_ocr_confidence: {ocr['mean_conf']:.2f}")
    return {"answers": answers, "marks": marks, "flags": flags, "roll_hint": find_roll_number(rows)}
