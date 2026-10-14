"""
backend/grader.py -- grade papers with an interchangeable "engine".

Engine specs (what you pick on the Rubric & Fine-tune page):
  ollama:<model>   any Ollama model, e.g. ollama:llama3.1:8b   (zero-shot baseline)
  hf:base          the fine-tuning base model, adapter OFF       (fair baseline)
  hf:finetuned     base model + your LoRA adapter                (the fine-tuned model)
  demo             similarity-based stand-in (AAE_MODE=demo only)

A grader only ever receives the STUDENT'S ANSWERS (folder 1). It never sees
the teacher's marks for the paper it is grading.
"""
import json

import config
from backend import exam_schema as es
from backend import prompts, text_sim


class Engine:
    name = "engine"
    with_rationale = False
    use_rubric = True

    def grade(self, messages, ctx):
        raise NotImplementedError


class OllamaEngine(Engine):
    with_rationale = True

    def __init__(self, model):
        self.model = model
        self.name = f"ollama:{model}"

    def grade(self, messages, ctx):
        from backend import ollama_client
        return ollama_client.chat(messages, model=self.model, json_mode=True, temperature=0.0)


class HFEngine(Engine):
    def __init__(self, finetuned=True):
        from backend import hf_infer
        if not hf_infer.stack_available():
            raise RuntimeError("torch / transformers / peft / bitsandbytes are not installed (README step 4).")
        if not hf_infer.adapter_ready():
            raise RuntimeError("No fine-tuned adapter yet - run the fine-tuning step first.")
        self.finetuned = finetuned
        self.name = "hf:finetuned" if finetuned else "hf:base"
        self.use_rubric = hf_infer.adapter_meta().get("use_rubric", True)
        self._hf = hf_infer

    def grade(self, messages, ctx):
        return self._hf.generate(messages, disable_adapter=not self.finetuned)


class DemoEngine(Engine):
    name = "demo"
    with_rationale = True

    def grade(self, messages, ctx):
        exam, qid, answers = ctx["exam"], ctx["qid"], ctx["answers"]
        q = es.get_question(exam, qid)
        model = es.model_answers_by_id(exam)
        sims = []
        for aid in es.question_answer_ids(exam, qid):
            txt = answers.get(aid, "")
            sims.append(text_sim.tfidf_scores(model.get(aid, ""), [txt])[0] if txt.strip() else 0.0)
        frac = min(1.0, (sum(sims) / max(1, len(sims))) * 1.15)
        return json.dumps({"marks": round((q["max_marks"] or 0) * frac * 2) / 2,
                           "rationale": "[DEMO] similarity to the model answer"})


def make_engine(spec):
    if spec == "demo":
        if not config.DEMO_MODE:
            raise ValueError("The demo engine is only available with AAE_MODE=demo.")
        return DemoEngine()
    if spec.startswith("ollama:"):
        return OllamaEngine(spec.split(":", 1)[1])
    if spec == "hf:base":
        return HFEngine(finetuned=False)
    if spec == "hf:finetuned":
        return HFEngine(finetuned=True)
    raise ValueError(f"Unknown engine '{spec}'")


def run_tag(spec):
    return spec.replace(":", "-").replace("/", "-").replace(" ", "")


def grade_paper(exam, answers, rubrics, engine):
    """-> {qid: {"marks": float|None, "rationale": str, "raw": str}}"""
    out = {}
    for q in exam["questions"]:
        aids = es.question_answer_ids(exam, q["id"])
        parts = {a: answers.get(a, "") for a in aids}
        rub = rubrics.get(q["id"]) if engine.use_rubric else None
        msgs = prompts.build_messages(exam, q["id"], parts, rub, with_rationale=engine.with_rationale)
        raw = engine.grade(msgs, {"exam": exam, "qid": q["id"], "answers": answers})
        marks, why = prompts.parse_grade(raw, q["max_marks"])
        rule = None
        if config.ENFORCE_ALL_PARTS_RULE and q["parts"] and any(not parts[a].strip() for a in aids):
            marks, rule = 0.0, "all-parts rule: a sub-question was left blank"
        out[q["id"]] = {"marks": marks, "rationale": rule or why, "raw": raw[:200]}
    return out
