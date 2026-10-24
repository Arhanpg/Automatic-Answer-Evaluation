"""
backend/sampling.py -- the AI chooses which papers become the labelled (training) set.

LABEL-FREE: it looks ONLY at folder 1 (the students' answers) and the model
answer key. It never reads folder 2 (the marks), exactly like a real
deployment where marks do not exist yet for the papers being chosen.

For every paper we measure
  * similarity of each answer to the model answer (TF-IDF, or Ollama embeddings),
  * completeness (how many sub-questions were actually answered),
and combine them into a quality proxy. Papers are ranked and cut into k equal
bands; the lowest and highest paper anchor the two ends, and each middle band
contributes its most representative paper. The result spans weak -> strong
students, which is what the rubric / fine-tuning step needs to see.
"""
import math

import config
from backend import exam_schema as es
from backend import text_sim
from backend.text_sim import tokenize

MIN_ANSWER_TOKENS = 2


def part_similarities(exam, answers_by_roll, backend="auto"):
    """-> ({roll: {aid: sim}}, backend_used)"""
    rolls = list(answers_by_roll)
    model = es.model_answers_by_id(exam)
    sims = {r: {} for r in rolls}
    used = "tfidf"
    embed = None
    if backend in ("auto", "embed") and not config.DEMO_MODE:
        try:
            from backend import ollama_client
            if ollama_client.ollama_available():
                embed = ollama_client.embed
        except Exception:
            embed = None
    for aid in es.answer_ids(exam):
        texts = [answers_by_roll[r].get(aid, "") for r in rolls]
        scores = None
        if embed is not None and model.get(aid):
            try:
                vecs = embed([model[aid]] + [t or " " for t in texts])
                scores = [text_sim.cosine_dense(vecs[0], v) if t.strip() else 0.0
                          for v, t in zip(vecs[1:], texts)]
                used = "ollama-embeddings"
            except Exception:
                scores = None
        if scores is None:
            scores = text_sim.tfidf_scores(model.get(aid, ""), texts)
        for r, s in zip(rolls, scores):
            sims[r][aid] = s
    return sims, used


def _weights(exam):
    w = {}
    for q in exam["questions"]:
        if q["parts"]:
            for p in q["parts"]:
                w[f"{q['id']}{p['id']}"] = p["marks"] or ((q["max_marks"] or 1) / len(q["parts"]))
        else:
            w[q["id"]] = q["max_marks"] or 1
    return w


def paper_features(exam, answers, sims):
    w = _weights(exam)
    ids = es.answer_ids(exam)
    answered = {a: len(tokenize(answers.get(a, ""))) >= MIN_ANSWER_TOKENS for a in ids}
    tw = sum(w.values()) or 1.0
    mean_sim = sum(w[a] * sims.get(a, 0.0) for a in ids) / tw
    completeness = sum(1 for a in ids if answered[a]) / max(1, len(ids))
    per_q = {}
    for q in exam["questions"]:
        qa = es.question_answer_ids(exam, q["id"])
        qw = sum(w[a] for a in qa) or 1.0
        per_q[q["id"]] = {
            "sim": sum(w[a] * sims.get(a, 0.0) for a in qa) / qw,
            "cov": sum(1 for a in qa if answered[a]) / max(1, len(qa)),
        }
    return {"mean_sim": mean_sim, "completeness": completeness, "per_q": per_q,
            "words": sum(len(tokenize(answers.get(a, ""))) for a in ids),
            "score": 0.7 * mean_sim + 0.3 * completeness}


def _vec(f, qids):
    return [f["per_q"][q]["sim"] for q in qids] + [f["per_q"][q]["cov"] for q in qids]


def select_training_papers(exam, answers_by_roll, flags_by_roll, k, backend="auto"):
    """-> (selected_rolls, details {roll: {...}}, chart_points, backend_used)"""
    sims, used = part_similarities(exam, answers_by_roll, backend)
    feats = {r: paper_features(exam, answers_by_roll[r], sims[r]) for r in answers_by_roll}

    def usable(r):
        bad = any(str(f).startswith("no_labels_found") for f in flags_by_roll.get(r, []))
        return not bad and feats[r]["words"] >= 5

    elig = [r for r in feats if usable(r)]
    if len(elig) < max(2, k):
        elig = list(feats)
    n = len(elig)
    k = max(1, min(k, n - 1 if n > 1 else 1))

    ranked = sorted(elig, key=lambda r: feats[r]["score"])
    qids = es.question_ids(exam)
    vecs = {r: _vec(feats[r], qids) for r in ranked}
    dim = len(next(iter(vecs.values())))
    mu = [sum(v[i] for v in vecs.values()) / n for i in range(dim)]
    sd = [math.sqrt(sum((v[i] - mu[i]) ** 2 for v in vecs.values()) / n) or 1.0 for i in range(dim)]
    z = {r: [(v[i] - mu[i]) / sd[i] for i in range(dim)] for r, v in vecs.items()}

    edges = [round(i * n / k) for i in range(k + 1)]
    selected, details = [], {}
    for b in range(k):
        band = ranked[edges[b]:edges[b + 1]] or ranked[min(edges[b], n - 1):min(edges[b], n - 1) + 1]
        if k > 1 and b == 0:
            pick, role = band[0], "weakest"
        elif k > 1 and b == k - 1:
            pick, role = band[-1], "strongest"
        else:
            cen = [sum(z[r][i] for r in band) / len(band) for i in range(dim)]
            pick = min(band, key=lambda r: sum((z[r][i] - cen[i]) ** 2 for i in range(dim)))
            role = "representative"
        if pick in selected:
            continue
        selected.append(pick)
        f = feats[pick]
        details[pick] = {"band": b + 1, "of": k, "role": role, "score": round(f["score"], 3),
                         "mean_sim": round(f["mean_sim"], 3), "completeness": round(f["completeness"], 2),
                         "words": f["words"]}
    chart = [{"roll": r, "score": round(feats[r]["score"], 3), "selected": r in selected} for r in ranked]
    return selected, details, chart, used
