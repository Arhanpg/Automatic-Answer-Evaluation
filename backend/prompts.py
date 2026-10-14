"""
backend/prompts.py -- ONE place that builds grading prompts.

The exact same builder is used to (a) create fine-tuning examples and
(b) grade papers at test time. If those two ever differ, the fine-tuned model
sees a format it was never trained on - so they must not.
"""
import json
import re

import config
from backend import exam_schema as es


def _fmt(x):
    if x is None:
        return "?"
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def _clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " ..."


def system_prompt(exam, with_rationale=False):
    rules = f"Exam rule: {exam['rules']} " if exam.get("rules") else ""
    if with_rationale:
        fmt = '{"marks": <number>, "rationale": "<one short sentence>"}'
    else:
        fmt = '{"marks": <number>}'
    return ("You are a university exam grader. Grade ONE question at a time, following the marking "
            "scheme and imitating this teacher's marking style as shown in the rubric. "
            f"{rules}Marks must be between 0 and the maximum, in steps of {config.GRADE_STEP}. "
            f"Reply with ONLY a JSON object: {fmt}")


def build_messages(exam, qid, student_parts, rubric=None, with_rationale=False):
    """student_parts: {answer_id: text} for the answer ids of THIS question."""
    q = es.get_question(exam, qid)
    lines = [f"Question {q['id']} [max {_fmt(q['max_marks'])} marks]"]
    if q["stem"]:
        lines.append(q["stem"])
    for p in q["parts"]:
        lines.append(f"({p['id']}) {p['text']} [{_fmt(p['marks'])} marks]")
    lines.append("")
    lines.append("Model answer:")
    if q["parts"]:
        for p in q["parts"]:
            lines.append(f"({p['id']}) {_clip(p['model_answer'], 500) or '-'}")
    else:
        lines.append(_clip(q["model_answer"], 700) or "-")
    if rubric:
        lines.append("")
        lines.append("Teacher's rubric:")
        for kp in rubric.get("key_points", []):
            lines.append(f"- {kp}")
        if rubric.get("marking_style_notes"):
            lines.append(f"Marking style: {rubric['marking_style_notes']}")
        for d in rubric.get("deduction_rules", []):
            lines.append(f"Deduct: {d}")
        st = rubric.get("marks_stats")
        if st and st.get("n"):
            lines.append(f"Teacher's marks on this question so far: min {_fmt(st['min'])}, "
                         f"max {_fmt(st['max'])}, average {st['mean']:.1f}")
    lines.append("")
    lines.append("Student answer:")
    blank = []
    aids = es.question_answer_ids(exam, qid)
    for aid in aids:
        text = (student_parts.get(aid) or "").strip()
        label = f"({aid[len(q['id']):]}) " if q["parts"] else ""
        if text:
            lines.append(f"{label}{_clip(text, config.MAX_PART_CHARS)}")
        else:
            lines.append(f"{label}(not answered)")
            blank.append(aid[len(q["id"]):] or q["id"])
    if blank and q["parts"]:
        lines.append(f"Parts left blank: {', '.join(blank)}")
    return [{"role": "system", "content": system_prompt(exam, with_rationale)},
            {"role": "user", "content": "\n".join(lines)}]


def target_json(marks):
    return json.dumps({"marks": marks})


def extract_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_grade(text, max_marks):
    """-> (marks|None, rationale). Never invents a mark: unparseable -> None."""
    data = extract_json(text)
    val, why = None, ""
    if isinstance(data, dict):
        for key in ("marks", "predicted_marks", "score"):
            if key in data:
                try:
                    val = float(data[key])
                except (TypeError, ValueError):
                    pass
                break
        why = str(data.get("rationale", ""))
    if val is None:
        m = re.search(r"(\d+(?:\.\d+)?)", text or "")
        if m and not data:
            val = float(m.group(1))
    if val is None:
        return None, why
    if max_marks is not None:
        val = max(0.0, min(float(max_marks), val))
    step = config.GRADE_STEP
    return round(val / step) * step, why
