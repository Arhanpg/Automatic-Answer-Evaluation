"""
backend/rubric_builder.py -- build a per-question rubric from the TRAINING papers.

Inputs : exam (question + model answer + marking scheme) and the training
         papers (student answers + the marks the teacher gave).
Output : data/rubrics/q<N>.json  with key points, marking-style notes,
         deduction rules and the teacher's real mark statistics.
Only training papers are used - the test papers are never looked at here.
"""
import json
import os

import config
from backend import exam_schema as es
from backend import prompts, text_sim

SYSTEM = ("You help build a grading rubric from a teacher's own marking. You get ONE exam question, "
          "its model answer, and several student answers together with the marks THIS teacher actually "
          "awarded. Infer what the teacher rewards and what they penalise. Reply with ONLY JSON: "
          '{"key_points": [short strings], "marking_style_notes": "1-3 sentences about strictness and '
          'partial credit", "deduction_rules": [short strings]}')


def _examples(exam, qid, train_data, limit=12):
    ex = []
    for rec in train_data:
        m = rec["marks"].get(qid)
        if m is None:
            continue
        ex.append({"roll": rec["roll"], "marks": m,
                   "parts": {a: rec["answers"].get(a, "") for a in es.question_answer_ids(exam, qid)}})
    ex.sort(key=lambda e: -e["marks"])
    if len(ex) > limit:      # keep the spread: best, worst and evenly spaced between
        idx = sorted({round(i * (len(ex) - 1) / (limit - 1)) for i in range(limit)})
        ex = [ex[i] for i in idx]
    return ex


def _stats(examples):
    ms = [e["marks"] for e in examples]
    if not ms:
        return {"n": 0}
    return {"n": len(ms), "min": min(ms), "max": max(ms), "mean": sum(ms) / len(ms)}


def _user_prompt(exam, qid, examples):
    q = es.get_question(exam, qid)
    base = prompts.build_messages(exam, qid, {}, rubric=None)[1]["content"].split("Student answer:")[0]
    out = [base.strip(), "", "Student answers with the teacher's marks:"]
    for i, e in enumerate(examples, start=1):
        out.append(f"--- Example {i}: teacher gave {e['marks']:g}/{q['max_marks']:g}")
        for aid, text in e["parts"].items():
            label = f"({aid[len(qid):]}) " if q["parts"] else ""
            out.append(f"{label}{(text or '(not answered)')[:350]}")
    return "\n".join(out)


def _demo_rubric(exam, qid):
    q = es.get_question(exam, qid)
    texts = [p["model_answer"] for p in q["parts"]] or [q["model_answer"]]
    pts = []
    for t in texts:
        first = (t or "").strip().split("\n")[0][:140]
        if first:
            pts.append(first)
    return {"key_points": pts, "marking_style_notes": "[DEMO] rubric derived from the model answer only.",
            "deduction_rules": []}


def build_rubric(exam, qid, train_data, chat=None):
    q = es.get_question(exam, qid)
    examples = _examples(exam, qid, train_data)
    if config.DEMO_MODE:
        parsed = _demo_rubric(exam, qid)
    else:
        if chat is None:
            from backend import ollama_client
            chat = lambda msgs: ollama_client.chat(msgs, json_mode=True, temperature=0.1)  # noqa: E731
        reply = chat([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": _user_prompt(exam, qid, examples)}])
        parsed = prompts.extract_json(reply) or {"key_points": [], "marking_style_notes": reply[:400],
                                                 "deduction_rules": []}
    return {
        "question_id": qid,
        "max_marks": q["max_marks"],
        "scheme": [{"part": p["id"], "marks": p["marks"]} for p in q["parts"]],
        "key_points": [str(x) for x in parsed.get("key_points", [])][:10],
        "marking_style_notes": str(parsed.get("marking_style_notes", ""))[:600],
        "deduction_rules": [str(x) for x in parsed.get("deduction_rules", [])][:8],
        "marks_stats": _stats(examples),
        "examples_used": len(examples),
    }


def build_all(exam, train_data, chat=None, progress=None):
    rubrics = {}
    qs = es.question_ids(exam)
    for i, qid in enumerate(qs, start=1):
        rubrics[qid] = build_rubric(exam, qid, train_data, chat)
        with open(os.path.join(config.RUBRICS_DIR, f"q{qid}.json"), "w", encoding="utf-8") as f:
            json.dump(rubrics[qid], f, indent=2, ensure_ascii=False)
        if progress:
            progress(i, len(qs), f"Rubric for Q{qid}")
    return rubrics


def load_all(exam):
    out = {}
    for qid in es.question_ids(exam):
        p = os.path.join(config.RUBRICS_DIR, f"q{qid}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[qid] = json.load(f)
    return out
