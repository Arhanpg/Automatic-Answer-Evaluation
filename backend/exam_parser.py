"""
backend/exam_parser.py -- question paper + model answer key  ->  exam draft.

1. read_lines(): text layer for digital PDFs (exact), PaddleOCR for scans/images.
2. parse_question_paper(): numbers, sub-questions (a, b, c...), marks ("4X2=8",
   "2+2+2", "3", "[3 marks]"), Full Marks, exam rules.
3. attach_model_answers(): the key repeats each question and then answers it.
   We align the key to the paper, drop the repeated question text and keep the
   answer. If the key's question text differs from the paper's, you get a warning
   (this actually happens in the sample paper you sent: Q3(c)).

The teacher reviews/edits the result on the Exam Setup page before approving.
"""
import os
import re
from difflib import SequenceMatcher

import config
from backend import pdf_utils
from backend.exam_schema import normalize_exam, validate_exam

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)

Q_RE = re.compile(r"^(?:Q(?:uestion)?\.?\s*)?(\d{1,2})\s*[\.\)](?!\d)\s*(.*)$", re.I)
SUB_RE = re.compile(r"^[\(\[]?\s*([A-Za-z])\s*[\.\)\]]\s*(.*)$")

_MUL = r"(\d{1,2})\s*[xX\u00d7*]\s*(\d{1,2})\s*=\s*(\d{1,3})"
_PLUS = r"(\d{1,2}(?:\s*\+\s*\d{1,2})+)(?:\s*=\s*(\d{1,3}))?"
MUL_ONLY = re.compile(rf"^\s*{_MUL}\s*$")
PLUS_ONLY = re.compile(rf"^\s*{_PLUS}\s*$")
SINGLE_ONLY = re.compile(r"^\s*[\(\[]?\s*(\d{1,2}(?:\.\d)?)\s*(?:marks?|m)?\s*[\)\]]?\s*$", re.I)
MUL_TRAIL = re.compile(rf"\s+{_MUL}\s*$")
PLUS_TRAIL = re.compile(rf"\s+{_PLUS}\s*$")
BRACKET_TRAIL = re.compile(r"\s*[\(\[]\s*(\d{1,2}(?:\.\d)?)\s*marks?\s*[\)\]]\s*$", re.I)
BARE_TRAIL = re.compile(r"(?<=[A-Za-z\.\?\:\)\u201d\u2019\"'])\s+(\d{1,2})\s*$")


def clean_line(s):
    s = s.translate(_ZERO_WIDTH).replace("\u00a0", " ").replace("\f", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _natural(s):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def read_lines(paths, force_ocr=False):
    """Returns (lines, method) where method is 'text_layer', 'ocr' or both."""
    from backend import ocr_engine
    lines, methods = [], set()
    for p in sorted(paths, key=lambda x: _natural(os.path.basename(x))):
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            if config.PREFER_PDF_TEXT_LAYER and not force_ocr and pdf_utils.has_text_layer(p):
                for page_text in pdf_utils.pdf_text_pages(p):
                    lines.extend(page_text.splitlines())
                methods.add("text_layer")
                continue
            images = pdf_utils.render_pdf(p, config.OCR_DPI)
        else:
            images = [pdf_utils.load_image(p)]
        lines.extend(ocr_engine.ocr_printed_lines(images))
        methods.add("ocr")
    cleaned = [clean_line(l) for l in lines]
    return [l for l in cleaned if l], "+".join(sorted(methods)) or "none"


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------
def split_marks_expr(line, allow_single=True):
    """-> (line_without_marks, expr|None). expr = {'kind': mul|plus|single, ...}"""
    m = MUL_ONLY.match(line)
    if m:
        return "", {"kind": "mul", "n": int(m[1]), "each": int(m[2]), "total": int(m[3])}
    m = PLUS_ONLY.match(line)
    if m:
        nums = [int(x) for x in re.split(r"\s*\+\s*", m[1])]
        return "", {"kind": "plus", "nums": nums, "total": int(m[2]) if m[2] else sum(nums)}
    if allow_single and SINGLE_ONLY.match(line):
        return "", {"kind": "single", "value": float(SINGLE_ONLY.match(line)[1])}
    m = MUL_TRAIL.search(line)
    if m:
        return line[:m.start()].strip(), {"kind": "mul", "n": int(m[1]), "each": int(m[2]), "total": int(m[3])}
    m = PLUS_TRAIL.search(line)
    if m:
        nums = [int(x) for x in re.split(r"\s*\+\s*", m[1])]
        return line[:m.start()].strip(), {"kind": "plus", "nums": nums, "total": int(m[2]) if m[2] else sum(nums)}
    if allow_single:
        m = BRACKET_TRAIL.search(line)
        if m:
            return line[:m.start()].strip(), {"kind": "single", "value": float(m[1])}
        m = BARE_TRAIL.search(line)
        if m:
            return line[:m.start()].strip(), {"kind": "single", "value": float(m[1])}
    return line, None


def parse_structure(lines, allow_single=True, full_marks=None):
    """Split into header lines + questions. Question numbers must run 1,2,3...
    and sub-question letters a,b,c... (so 'i.', 'ii.' data lists are not mistaken
    for sub-questions)."""
    header, questions = [], []
    cur, expected = None, "a"
    for raw in lines:
        line = clean_line(raw)
        if not line:
            continue
        mq = Q_RE.match(line)
        if mq and int(mq.group(1)) == len(questions) + 1:
            cur = {"num": mq.group(1), "stem_lines": [], "parts": [], "marks_exprs": []}
            questions.append(cur)
            expected = "a"
            rest, expr = split_marks_expr(mq.group(2).strip(), allow_single)
            if expr:
                cur["marks_exprs"].append(expr)
            if rest:
                cur["stem_lines"].append(rest)
            continue
        if cur is None:
            header.append(line)
            continue
        ms = SUB_RE.match(line)
        if ms and ms.group(1).lower() == expected:
            rest, expr = split_marks_expr(ms.group(2).strip(), allow_single)
            if expr:
                cur["marks_exprs"].append({**expr, "part": expected})
            cur["parts"].append({"id": expected, "lines": [rest] if rest else []})
            expected = chr(ord(expected) + 1)
            continue
        rest, expr = split_marks_expr(line, allow_single)
        if expr and expr["kind"] == "single" and full_marks and expr["value"] > full_marks:
            rest, expr = line, None
        if expr:
            cur["marks_exprs"].append(expr)
        if rest:
            (cur["parts"][-1]["lines"] if cur["parts"] else cur["stem_lines"]).append(rest)
    return header, questions


def combine_marks(exprs, n_parts):
    total, per = None, None
    for e in exprs:
        if e["kind"] == "mul":
            total = e["total"]
            if e["n"] == n_parts:
                per = [e["each"]] * n_parts
        elif e["kind"] == "plus":
            nums = e["nums"]
            total = e["total"]
            if n_parts and len(nums) == n_parts:
                per = nums
    if total is None:
        singles = [e["value"] for e in exprs if e["kind"] == "single" and "part" not in e]
        if singles:
            total = singles[-1]
    return total, per


_META_RE = re.compile(r"^(timing|time\b|duration|full\s*marks|max(imum)?\s*marks|\d+(st|nd|rd|th)\s+sem)", re.I)


def parse_header(header):
    institute = header[0] if header else ""
    full, idx_full = None, None
    for i, l in enumerate(header):
        m = re.search(r"(?:full|max(?:imum)?)\s*marks?\s*[:\-]?\s*(\d+)", l, re.I)
        if m:
            full, idx_full = int(m.group(1)), i
    first_meta = next((i for i, l in enumerate(header) if i > 0 and _META_RE.match(l)), len(header))
    title = " \u2014 ".join(header[1:first_meta])
    if idx_full is not None:
        rules_lines = header[idx_full + 1:]
    else:
        start = next((i for i, l in enumerate(header)
                      if re.match(r"^(no marks|note|instruction)", l, re.I)), len(header))
        rules_lines = header[start:]
    return {"institute": institute, "title": title, "total_marks": full, "rules": " ".join(rules_lines)}


def parse_question_paper(lines):
    probe = None
    for l in lines:
        m = re.search(r"full\s*marks?\s*[:\-]?\s*(\d+)", l, re.I)
        if m:
            probe = int(m.group(1))
    header, qs = parse_structure(lines, allow_single=True, full_marks=probe)
    meta = parse_header(header)
    questions = []
    for q in qs:
        parts = [{"id": p["id"], "text": " ".join(p["lines"]).strip(), "marks": None, "model_answer": ""}
                 for p in q["parts"]]
        total, per = combine_marks(q["marks_exprs"], len(parts))
        if per:
            for p, m in zip(parts, per):
                p["marks"] = m
        if total is None and per:
            total = sum(per)
        questions.append({"id": q["num"], "stem": "\n".join(q["stem_lines"]).strip(),
                          "max_marks": total, "parts": parts, "model_answer": ""})
    exam = {**meta, "questions": questions, "warnings": []}
    return normalize_exam(exam)


# ---------------------------------------------------------------------------
# model answer alignment
# ---------------------------------------------------------------------------
def _is_marks_only(line):
    return bool(MUL_ONLY.match(line) or PLUS_ONLY.match(line) or SINGLE_ONLY.match(line))


def _strip_trailing_marks(line):
    """'... tokens. 3' -> '... tokens.'  (the key often prints the marks at the end of the question)"""
    stripped = re.sub(r"(?<=[A-Za-z\.\?\:\)\u201d\u2019\"'])\s+\d{1,2}$", "", line)
    return stripped if len(tokens(stripped)) >= 3 else line


def split_question_answer(lines, paper_text):
    """The key repeats the question, then answers. Return (question_text, answer_lines)."""
    if not lines:
        return "", []
    tq = set(tokens(paper_text))
    lines = [_strip_trailing_marks(l) for l in lines]
    qlines, i = [lines[0]], 1
    while i < len(lines):
        t = tokens(lines[i])
        if t and sum(1 for w in t if w in tq) / len(t) >= 0.8 and not _is_marks_only(lines[i]):
            qlines.append(lines[i])
            i += 1
        else:
            break
    rest = list(lines[i:])
    while rest and _is_marks_only(rest[0]):
        rest.pop(0)
    return " ".join(qlines), rest


def _norm(s):
    return " ".join(tokens(s))


def attach_model_answers(exam, key_lines):
    """Fill model_answer fields from the key. Returns a list of warnings."""
    warnings = []
    _, kqs = parse_structure(key_lines, allow_single=False)
    kmap = {q["num"]: q for q in kqs}
    for q in exam["questions"]:
        kq = kmap.get(q["id"])
        if kq is None:
            warnings.append(f"Q{q['id']}: not found in the model answer key.")
            continue
        if q["parts"]:
            kparts = {p["id"]: p for p in kq["parts"]}
            if not kparts:
                q["model_answer"] = "\n".join(kq["stem_lines"])
                warnings.append(f"Q{q['id']}: the key has no (a)/(b)... sub-answers; "
                                f"its whole answer was stored at question level.")
                continue
            for p in q["parts"]:
                kp = kparts.get(p["id"])
                if kp is None:
                    warnings.append(f"Q{q['id']}({p['id']}): missing in the model answer key.")
                    continue
                qtext, ans = split_question_answer(kp["lines"], p["text"])
                p["model_answer"] = "\n".join(ans).strip()
                sim = SequenceMatcher(None, _norm(qtext), _norm(p["text"])).ratio()
                if sim < 0.6:
                    warnings.append(
                        f"Q{q['id']}({p['id']}): the key's question text (\"{qtext[:70]}\") does not match "
                        f"the question paper (\"{p['text'][:70]}\"). The answer stored is the one written "
                        f"under the key's version - please check it answers the paper's question.")
            for pid in sorted(set(kparts) - {p["id"] for p in q["parts"]}):
                warnings.append(f"Q{q['id']}({pid}): present in the key but not in the question paper.")
        else:
            qtext, ans = split_question_answer(kq["stem_lines"], q["stem"])
            q["model_answer"] = "\n".join(ans).strip()
    return warnings


def build_exam_draft(paper_paths, key_paths, force_ocr=False):
    paper_lines, m1 = read_lines(paper_paths, force_ocr)
    exam = parse_question_paper(paper_lines)
    warnings = []
    m2 = "none"
    if key_paths:
        key_lines, m2 = read_lines(key_paths, force_ocr)
        warnings += attach_model_answers(exam, key_lines)
    exam = normalize_exam(exam)
    exam["read_warnings"] = warnings          # found while reading / aligning the two documents
    exam["warnings"] = validate_exam(exam)     # live checks, recomputed on every approve
    exam["source"] = {"question_paper": m1, "model_answer": m2}
    return exam
