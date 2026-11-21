"""
research_v3/phase9_ocr_benchmarking/metrics.py

Comprehensive assessment-critical evaluation metrics for handwriting recognition:
1. Character Error Rate (CER) and Word Error Rate (WER) via Levenshtein edit distance.
2. Exact Response Match (binary 0/1 correctness).
3. Mathematical token precision, recall, and F1 score.
4. Numeric constant accuracy.
5. Formula and structural syntax preservation.
6. Omission detection accuracy.
"""

import re
from typing import Dict, List, Set, Tuple, Any
from rapidfuzz.distance import Levenshtein

MATH_SYMBOLS = {
    "+", "-", "*", "/", "=", "^", "<", ">", "<=", ">=", "!=",
    "softmax", "dot", "exp", "sum", "argmax", "max", "min",
    "p(", "q(", "k1", "k2", "v1", "v2", "q.k", "q·k",
    "<s>", "</s>", "<s", "/s>", "<s >", "< /s >",
    "laplace", "smoothing", "vocab", "alpha", "beta", "lambda"
}

NUMERIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


def compute_cer(ref: str, hyp: str) -> float:
    """Computes Character Error Rate. Note: Can exceed 1.0 if insertions exceed ref length."""
    if not ref and not hyp:
        return 0.0
    if not ref:
        return float(len(hyp))
    dist = Levenshtein.distance(ref, hyp)
    return dist / len(ref)


def compute_wer(ref: str, hyp: str) -> float:
    """Computes Word Error Rate. Note: Can exceed 1.0 if insertions exceed ref length."""
    r_words = ref.split()
    h_words = hyp.split()
    if not r_words and not h_words:
        return 0.0
    if not r_words:
        return float(len(h_words))
    dist = Levenshtein.distance(r_words, h_words)
    return dist / len(r_words)


def compute_exact_match(ref: str, hyp: str) -> float:
    """Binary exact response match indicator after whitespace trimming."""
    return 1.0 if ref.strip().lower() == hyp.strip().lower() else 0.0


def extract_math_tokens(text: str) -> List[str]:
    """Extracts mathematical symbols, operators, and functions from text."""
    lower = text.lower()
    found = []
    # Check operator symbols and key terms
    for sym in MATH_SYMBOLS:
        count = lower.count(sym)
        if count > 0:
            found.extend([sym] * count)
    return found


def compute_math_metrics(ref: str, hyp: str) -> Dict[str, float]:
    """Computes Precision, Recall, and F1 for mathematical symbols."""
    ref_tokens = extract_math_tokens(ref)
    hyp_tokens = extract_math_tokens(hyp)

    if not ref_tokens and not hyp_tokens:
        return {"math_precision": 1.0, "math_recall": 1.0, "math_f1": 1.0, "ref_math_count": 0}
    if not ref_tokens:
        return {"math_precision": 0.0, "math_recall": 1.0, "math_f1": 0.0, "ref_math_count": 0}
    if not hyp_tokens:
        return {"math_precision": 0.0, "math_recall": 0.0, "math_f1": 0.0, "ref_math_count": len(ref_tokens)}

    # Multiset intersection
    ref_copy = list(ref_tokens)
    matched = 0
    for tok in hyp_tokens:
        if tok in ref_copy:
            matched += 1
            ref_copy.remove(tok)

    precision = matched / len(hyp_tokens) if hyp_tokens else 0.0
    recall = matched / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "math_precision": round(precision, 4),
        "math_recall": round(recall, 4),
        "math_f1": round(f1, 4),
        "ref_math_count": len(ref_tokens)
    }


def extract_numbers(text: str) -> List[str]:
    """Extracts numeric digit sequences and floating-point constants."""
    return NUMERIC_PATTERN.findall(text)


def compute_numeric_metrics(ref: str, hyp: str) -> Dict[str, float]:
    """Computes Precision, Recall, and F1 for numeric constants."""
    ref_nums = extract_numbers(ref)
    hyp_nums = extract_numbers(hyp)

    if not ref_nums and not hyp_nums:
        return {"num_precision": 1.0, "num_recall": 1.0, "num_f1": 1.0, "ref_num_count": 0}
    if not ref_nums:
        return {"num_precision": 0.0, "num_recall": 1.0, "num_f1": 0.0, "ref_num_count": 0}
    if not hyp_nums:
        return {"num_precision": 0.0, "num_recall": 0.0, "num_f1": 0.0, "ref_num_count": len(ref_nums)}

    ref_copy = list(ref_nums)
    matched = 0
    for num in hyp_nums:
        if num in ref_copy:
            matched += 1
            ref_copy.remove(num)

    precision = matched / len(hyp_nums) if hyp_nums else 0.0
    recall = matched / len(ref_nums) if ref_nums else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "num_precision": round(precision, 4),
        "num_recall": round(recall, 4),
        "num_f1": round(f1, 4),
        "ref_num_count": len(ref_nums)
    }


def evaluate_prediction(ref: str, hyp: str, is_omitted: bool) -> Dict[str, Any]:
    """Comprehensive evaluation dictionary for a single answer prediction."""
    if is_omitted:
        # Correct omission detection means predicting blank or near-blank text
        detected_as_omitted = (len(hyp.strip()) == 0)
        return {
            "cer": 0.0 if detected_as_omitted else 1.0,
            "wer": 0.0 if detected_as_omitted else 1.0,
            "exact_match": 1.0 if detected_as_omitted else 0.0,
            "omission_correct": 1.0 if detected_as_omitted else 0.0,
            "math_precision": 1.0,
            "math_recall": 1.0,
            "math_f1": 1.0,
            "num_precision": 1.0,
            "num_recall": 1.0,
            "num_f1": 1.0
        }

    cer = compute_cer(ref, hyp)
    wer = compute_wer(ref, hyp)
    em = compute_exact_match(ref, hyp)
    math_eval = compute_math_metrics(ref, hyp)
    num_eval = compute_numeric_metrics(ref, hyp)

    res = {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "exact_match": em,
        "omission_correct": 0.0 if len(hyp.strip()) == 0 else 1.0,  # penalized if falsely predicted blank
    }
    res.update(math_eval)
    res.update(num_eval)
    return res
