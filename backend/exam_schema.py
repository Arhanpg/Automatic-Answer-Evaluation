"""
backend/exam_schema.py -- the exam definition everything else is built on.

exam.json (approved by the teacher on the Exam Setup page):
{
  "title": "...", "institute": "...", "total_marks": 20, "rules": "No marks will be ...",
  "questions": [
    {"id": "1", "stem": "Answer the following ...", "max_marks": 8,
     "parts": [{"id": "a", "text": "What is ...?", "marks": 2, "model_answer": "..."}],
     "model_answer": ""},                       # used only when a question has no parts
    {"id": "4", "stem": "For a training corpus ...", "max_marks": 3, "parts": [],
     "model_answer": "With <S> and </S> ..."}
  ]
}
Answer ids used in the student-answer JSONs: "1a", "1b", ..., "2a", ..., "4".
Marks ids (teacher-marks JSONs) are question numbers: "1", "2", "3", "4".
"""
import json
import os

import config


def exam_exists():
    return os.path.exists(config.EXAM_JSON)


def load_exam(path=None):
    path = path or config.EXAM_JSON
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_exam(exam, path=None):
    path = path or config.EXAM_JSON
    exam = normalize_exam(exam)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exam, f, indent=2, ensure_ascii=False)
    return exam


def _num(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else f


def normalize_exam(exam):
    """Coerce types and fill derived values. Safe to call repeatedly."""
    exam = dict(exam)
    qs = []
    for i, q in enumerate(exam.get("questions", []), start=1):
        q = dict(q)
        q["id"] = str(q.get("id") or i).strip()
        q["stem"] = (q.get("stem") or "").strip()
        parts = []
        for j, p in enumerate(q.get("parts") or []):
            p = dict(p)
            p["id"] = (str(p.get("id") or chr(ord("a") + j)).strip().lower())
            p["text"] = (p.get("text") or "").strip()
            p["marks"] = _num(p.get("marks"))
            p["model_answer"] = (p.get("model_answer") or "").strip()
            parts.append(p)
        q["parts"] = parts
        q["model_answer"] = (q.get("model_answer") or "").strip()
        mm = _num(q.get("max_marks"))
        if mm is None and parts and all(p["marks"] is not None for p in parts):
            mm = _num(sum(p["marks"] for p in parts))
        q["max_marks"] = mm
        qs.append(q)
    exam["questions"] = qs
    exam["total_marks"] = _num(exam.get("total_marks"))
    exam["rules"] = (exam.get("rules") or "").strip()
    exam["title"] = (exam.get("title") or "").strip()
    exam["institute"] = (exam.get("institute") or "").strip()
    exam.setdefault("warnings", [])
    exam["read_warnings"] = list(exam.get("read_warnings") or [])
    return exam


def question_ids(exam):
    return [q["id"] for q in exam["questions"]]


def answer_ids(exam):
    ids = []
    for q in exam["questions"]:
        if q["parts"]:
            ids += [f"{q['id']}{p['id']}" for p in q["parts"]]
        else:
            ids.append(q["id"])
    return ids


def get_question(exam, qid):
    for q in exam["questions"]:
        if q["id"] == str(qid):
            return q
    return None


def answer_id_to_question(exam):
    m = {}
    for q in exam["questions"]:
        if q["parts"]:
            for p in q["parts"]:
                m[f"{q['id']}{p['id']}"] = q["id"]
        else:
            m[q["id"]] = q["id"]
    return m


def question_answer_ids(exam, qid):
    q = get_question(exam, qid)
    if q is None:
        return []
    return [f"{q['id']}{p['id']}" for p in q["parts"]] if q["parts"] else [q["id"]]


def max_marks(exam, qid):
    q = get_question(exam, qid)
    return q["max_marks"] if q else None


def total_marks(exam):
    vals = [q["max_marks"] for q in exam["questions"]]
    return sum(v for v in vals if v is not None)


def model_answers_by_id(exam):
    out = {}
    for q in exam["questions"]:
        if q["parts"]:
            for p in q["parts"]:
                out[f"{q['id']}{p['id']}"] = p["model_answer"]
        else:
            out[q["id"]] = q["model_answer"]
    return out


def validate_exam(exam):
    """Human-readable problems the teacher should look at before approving."""
    w = []
    if not exam.get("questions"):
        return ["No questions were found in the question paper."]
    for q in exam["questions"]:
        if q["max_marks"] is None:
            w.append(f"Q{q['id']}: marks not detected - please enter them.")
        for p in q["parts"]:
            if not p["model_answer"]:
                w.append(f"Q{q['id']}({p['id']}): no model answer found.")
        if not q["parts"] and not q["model_answer"]:
            w.append(f"Q{q['id']}: no model answer found.")
        if q["parts"] and all(p["marks"] is not None for p in q["parts"]) and q["max_marks"] is not None:
            s = sum(p["marks"] for p in q["parts"])
            if abs(s - q["max_marks"]) > 1e-9:
                w.append(f"Q{q['id']}: part marks add up to {s} but the question total is {q['max_marks']}.")
    tm, fm = total_marks(exam), exam.get("total_marks")
    if fm is not None and abs(tm - fm) > 1e-9:
        w.append(f"Question marks add up to {tm} but the paper says Full Marks = {fm}.")
    return w
