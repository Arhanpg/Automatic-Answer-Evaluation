"""
backend/demo_data.py -- synthetic students for AAE_MODE=demo ONLY.

Builds fake-but-plausible answers by degrading the model answers (dropping
words, leaving parts blank) and marks that follow the same quality, so the
selection / rubric / evaluation screens have something sensible to show.
Nothing here is ever used in real mode.
"""
import random

from backend import exam_schema as es


def _round_half(x):
    return round(x * 2) / 2


def make_students(exam, n=30, seed=7):
    rnd = random.Random(seed)
    model = es.model_answers_by_id(exam)
    out = []
    for i in range(1, n + 1):
        quality = min(1.0, max(0.05, rnd.betavariate(2.2, 1.6)))
        answers, marks, flags = {}, {}, []
        for q in exam["questions"]:
            fracs = []
            for aid in es.question_answer_ids(exam, q["id"]):
                words = (model.get(aid) or "").split()
                if not words or rnd.random() < (1 - quality) * 0.25:
                    answers[aid] = ""
                    fracs.append(0.0)
                    continue
                keep = min(1.0, max(0.1, quality + rnd.uniform(-0.2, 0.2)))
                kept = [w for w in words if rnd.random() < keep] or words[:2]
                answers[aid] = " ".join(kept)
                fracs.append(len(kept) / len(words))
            qq = min(1.0, max(0.0, sum(fracs) / max(1, len(fracs)) + rnd.uniform(-0.08, 0.08)))
            marks[q["id"]] = _round_half(qq * (q["max_marks"] or 0))
        if rnd.random() < 0.1:
            flags.append("marks_Q2: ambiguous (several numbers)")
        out.append({"roll": f"demo{i}", "answers": answers, "marks": marks, "flags": flags})
    return out
