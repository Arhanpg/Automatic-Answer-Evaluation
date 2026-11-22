"""
research_v4/model_comparison/src/text_normalization.py

Text Normalization and Exact Longest Common Subsequence (ROUGE-L) Utilities:
1. Normalizes text while strictly preserving mathematical symbols (+, -, *, /, ^, _, =, <, >, ., %).
2. Computes exact DP-based Longest Common Subsequence (LCS) for genuine ROUGE-L F1, precision, and recall.
"""

import re
from typing import List, Tuple, Set


def clean_tokens(text: str, preserve_math: bool = True) -> List[str]:
    """Tokenizes text into words and mathematical operator tokens."""
    if not text:
        return []
    s = str(text).strip().lower()
    if preserve_math:
        # Match alphanumeric words, numbers (including decimals), or math symbols
        tokens = re.findall(r"[a-z]+|[0-9]+(?:\.[0-9]+)?|[-+*/^=<>_\[\](){}%\\]", s)
    else:
        tokens = re.findall(r"[a-z0-9]+", s)
    return tokens


def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces and strips margins."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def exact_lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """
    Computes exact Longest Common Subsequence (LCS) length via dynamic programming.
    Time complexity: O(|A| * |B|), Space complexity: O(min(|A|, |B|)).
    """
    if not seq_a or not seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    m, n = len(seq_a), len(seq_b)
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_true_rouge_l(student_text: str, reference_text: str) -> Tuple[float, float, float]:
    """
    Computes true ROUGE-L (LCS-based Precision, Recall, and F1).
    Returns: (precision, recall, f1)
    """
    s_toks = clean_tokens(student_text)
    r_toks = clean_tokens(reference_text)

    if not s_toks or not r_toks:
        return 0.0, 0.0, 0.0

    lcs = exact_lcs_length(s_toks, r_toks)
    prec = lcs / len(s_toks)
    rec = lcs / len(r_toks)

    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * prec * rec / (prec + rec)

    return float(prec), float(rec), float(f1)
