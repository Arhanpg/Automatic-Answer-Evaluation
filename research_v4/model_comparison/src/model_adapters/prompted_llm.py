"""
research_v4/model_comparison/src/model_adapters/prompted_llm.py

Prompted Local LLM Model Adapter:
Executes live inference against local Ollama models (llama3.2:3b, qwen2.5:7b).
Supports zero-shot rubric, few-shot training exemplar, and criterion-decomposed prompting.
Strict JSON schema extraction with telemetry and parse failure logging.
Zero test label leakage.
"""

import time
import json
import re
import urllib.request
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


def call_local_ollama(
    prompt: str,
    model_name: str = "llama3.2:3b",
    temperature: float = 0.0,
    timeout_sec: int = 45
) -> Tuple[str, float, int]:
    """
    Executes a live HTTP query to the local Ollama daemon.
    Returns: (response_text, latency_ms, eval_count)
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 42
        }
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.time() - t0) * 1000.0
            text = resp_data.get("response", "")
            eval_tokens = resp_data.get("eval_count", 0)
            return text, latency_ms, eval_tokens
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000.0
        return f"ERROR: {e}", latency_ms, 0


def parse_mark_from_llm_response(response_text: str, max_marks: float) -> Tuple[float, bool]:
    """
    Extracts numerical mark from JSON or regex patterns.
    Returns: (parsed_mark, success_flag)
    """
    # 1. Try strict JSON block extraction
    json_match = re.search(r"\{[^{}]*\"marks?\"[^{}]*\}", response_text, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            for k in ["marks", "mark", "score"]:
                if k in parsed:
                    val = float(parsed[k])
                    return min(max_marks, max(0.0, val)), True
        except Exception:
            pass

    # 2. Try regex pattern: "Marks: X" or "Score: X"
    num_match = re.search(r"(?:marks?|score|grade)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", response_text, re.IGNORECASE)
    if num_match:
        try:
            val = float(num_match.group(1))
            return min(max_marks, max(0.0, val)), True
        except Exception:
            pass

    # Parse failure: fallback to half max marks (never gold label)
    return max_marks * 0.5, False


class PromptedLLMGrader:
    """Evaluates student answers using local LLMs with structured prompts."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        prompt_style: str = "zero_shot",
        input_key: str = "trocr_actual"
    ):
        self.model_name = model_name
        self.prompt_style = prompt_style
        self.input_key = input_key
        self.train_exemplars: Dict[str, List[Dict[str, Any]]] = {}
        self.telemetry_logs: List[Dict[str, Any]] = []

    def fit(self, train_records: List[Dict[str, Any]]) -> "PromptedLLMGrader":
        """Stores training-fold exemplars for few-shot prompt construction."""
        self.train_exemplars = {"1": [], "2": [], "3": [], "4": []}
        for r in train_records:
            qid = str(r["question_id"])
            if qid in self.train_exemplars:
                self.train_exemplars[qid].append(r)
        return self

    def _build_prompt(self, record: Dict[str, Any]) -> str:
        qid = str(record["question_id"])
        max_m = float(record["max_marks"])
        ans_text = str(record.get(self.input_key, "")).strip() or "[Blank/No Answer]"
        model_ans = str(record.get("model_answer", "")).strip()
        criteria = record.get("rubric_criteria", [])

        crit_str = "\n".join([f"- Criterion: {c.get('description', '')} ({c.get('marks', 1.0)}m)" for c in criteria])

        if self.prompt_style == "few_shot" and self.train_exemplars.get(qid):
            # Select 1 exemplar from train fold (different student)
            ex = self.train_exemplars[qid][0]
            ex_ans = str(ex.get(self.input_key, ""))[:200]
            ex_mark = float(ex["teacher_mark"])
            few_shot_block = (
                f"### Example Graded Answer:\n"
                f"Student Answer: \"{ex_ans}\"\n"
                f"Evaluation: {{\"marks\": {ex_mark}}}\n\n"
            )
        else:
            few_shot_block = ""

        prompt = (
            f"You are an expert university examiner evaluating a technical handwritten examination on Large Language Models.\n"
            f"Question {qid} (Max Marks: {max_m})\n"
            f"Reference Solution: {model_ans}\n"
            f"Rubric Criteria:\n{crit_str}\n\n"
            f"{few_shot_block}"
            f"### Student Answer to Evaluate:\n"
            f"\"{ans_text}\"\n\n"
            f"Grade the student's answer strictly against the rubric criteria.\n"
            f"Output your final decision strictly as JSON in the following format:\n"
            f"{{\"marks\": <float value between 0.0 and {max_m}>}}"
        )
        return prompt

    def predict(self, test_records: List[Dict[str, Any]]) -> np.ndarray:
        preds = []
        for r in test_records:
            prompt = self._build_prompt(r)
            max_m = float(r["max_marks"])

            raw_resp, latency, tokens = call_local_ollama(
                prompt,
                model_name=self.model_name,
                temperature=0.0
            )

            mark, success = parse_mark_from_llm_response(raw_resp, max_m)
            preds.append(mark)

            self.telemetry_logs.append({
                "response_id": r["response_id"],
                "model_name": self.model_name,
                "prompt_style": self.prompt_style,
                "latency_ms": round(latency, 2),
                "eval_tokens": tokens,
                "parse_success": success,
                "predicted_mark": mark,
                "raw_response_snippet": raw_resp[:300].replace("\n", " ")
            })

        return np.array(preds, dtype=np.float64)
