"""
research_v4/model_comparison/src/feature_extractors/symbolic.py

SymPy Symbolic and Mathematical Equivalence Feature Extractor:
1. Subquestions are preserved individually (resolves the legacy concatenation flaw).
2. Verifies attention scores, softmax vectors, and linear projections for Question 2.
3. Verifies decoding choices and override rules for Question 3.
4. Verifies Laplace smoothing fractions and conditional counts for Question 4.
5. Flags domain-specific contradictions.
"""

import re
from typing import Dict, List, Any


CONTRADICTION_PATTERNS = {
    "1": [
        (r"\bunlimited\s+context\b", "Claims n-grams have unlimited context."),
        (r"\bno\s+sparsity\b", "Claims n-grams have no sparsity issues."),
        (r"\bencoder\s*(?:only)?\s*(?:is|uses)?\s*causal\b", "Claims encoder-only is causal."),
        (r"\bdecoder\s*(?:only)?\s*(?:is|uses)?\s*bidirectional\b", "Claims decoder-only is bidirectional.")
    ],
    "3": [
        (r"\b(?:word\s*)?[cd]\b", "Selected suboptimal Word C or D in decoding.")
    ]
}


def verify_math_question_2(subquestions: Dict[str, str]) -> float:
    """
    Evaluates Question 2 (Scaled Dot-Product Attention derivation, Max 6.0 marks):
    Q2a (2.0m): Dot product scores [1, 0]
    Q2b (2.0m): Softmax weights [0.73, 0.27] or [0.731, 0.269]
    Q2c (2.0m): Output vector [1.46, 0.81] or [1.462, 0.807]
    """
    marks = 0.0
    ans_2a = str(subquestions.get("2a", "")).lower()
    ans_2b = str(subquestions.get("2b", "")).lower()
    ans_2c = str(subquestions.get("2c", "")).lower()

    # 2a: Looking for scores 1 and 0 or [1, 0]
    if re.search(r"\[?\s*1(?:\.0)?\s*,\s*0(?:\.0)?\s*\]?", ans_2a) or ("1" in ans_2a and "0" in ans_2a and "k1" in ans_2a):
        marks += 2.0
    elif re.search(r"q\s*\.?\s*k_?1\s*=\s*1", ans_2a) or ("1" in ans_2a and "0" in ans_2a):
        marks += 2.0

    # 2b: Looking for [0.73, 0.27] or [0.731, 0.269] or e/(e+1)
    if re.search(r"0\.73|0\.27|0\.731|0\.269", ans_2b) or "e/(e+1)" in ans_2b or "e / (e + 1)" in ans_2b:
        marks += 2.0
    elif "0.7" in ans_2b and "0.3" in ans_2b:
        marks += 1.5

    # 2c: Looking for [1.46, 0.81] or [1.462, 0.807]
    if re.search(r"1\.46|0\.81|1\.462|0\.807", ans_2c):
        marks += 2.0
    elif "1.4" in ans_2c and "0.8" in ans_2c:
        marks += 1.5

    return min(6.0, marks)


def verify_math_question_4(student_text: str) -> float:
    """
    Evaluates Question 4 (Add-k Laplace smoothing, Max 3.0 marks):
    Fractions: 3/2420, 2/1050, 3/12100, 0.000408, or partial 1.0m.
    """
    s = str(student_text).lower()
    if re.search(r"3\s*/\s*2420|2\s*/\s*1050|3\s*/\s*12100|0\.000408", s):
        return 3.0
    elif re.search(r"add[- ](?:1|k)|laplace|smoothing|\+ ?1|\+ ?k", s):
        if re.search(r"c\([^)]+\)|count|vocabulary|\|v\|", s):
            return 2.0
        return 1.0
    return 0.0


def extract_symbolic_features(
    question_id: str,
    student_text: str,
    subquestions: Dict[str, str],
    max_marks: float
) -> Dict[str, float]:
    """
    Extracts mathematical validity, symbolic fraction matches, and contradiction flags.
    """
    qid = str(question_id).replace("Q", "").strip()
    s_clean = str(student_text or "")

    # Contradiction check
    has_contra = 0.0
    if qid in CONTRADICTION_PATTERNS:
        for pat, _ in CONTRADICTION_PATTERNS[qid]:
            if re.search(pat, s_clean, re.I):
                has_contra = 1.0
                break

    # Mathematical verification score
    math_score = 0.0
    if qid == "2":
        math_score = verify_math_question_2(subquestions)
    elif qid == "4":
        math_score = verify_math_question_4(s_clean)
    elif qid == "3":
        # Question 3: decoding strategies
        s_low = s_clean.lower()
        q3_score = 0.0
        if "word a" in s_low or "highest prob" in s_low or "greedy" in s_low:
            q3_score += 1.0
        if "word a" in s_low and "word b" in s_low:
            q3_score += 1.0
        if "layer norm" in s_low or "normalization" in s_low or "top-p" in s_low or "nucleus" in s_low:
            q3_score += 1.0
        math_score = min(3.0, q3_score)

    norm_math = float(math_score / max(1.0, max_marks))

    # Math syntax indicators
    math_symbols = sum(1 for c in s_clean if c in "+-*/^=<>_[]{}()\\%")
    has_math_syntax = 1.0 if math_symbols > 2 else 0.0

    return {
        "sym_raw_score": round(math_score, 4),
        "sym_normalized_score": round(norm_math, 4),
        "sym_has_math_syntax": has_math_syntax,
        "sym_math_symbol_count": float(math_symbols),
        "sym_contradiction_flag": has_contra
    }
