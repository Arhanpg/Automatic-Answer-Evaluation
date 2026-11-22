"""
research_v4/model_comparison/src/feature_extractors/lexical.py

Lexical and String Similarity Feature Extractor:
1. Exact match (case and space normalized).
2. Character-level normalized Levenshtein similarity.
3. Token-level normalized Levenshtein similarity.
4. Token Jaccard similarity (set intersection over union).
5. Token Precision, Recall, and F1.
6. Bigram & Trigram precision, recall, and overlap.
7. Exact DP Longest Common Subsequence ROUGE-L (Precision, Recall, F1).
8. Character and token length ratios.
"""

from typing import Dict, List, Any
import rapidfuzz
from ..text_normalization import clean_tokens, compute_true_rouge_l


def extract_lexical_features(student_text: str, reference_text: str) -> Dict[str, float]:
    """
    Computes fine-grained lexical and string alignment metrics between student and reference text.
    """
    s_clean = str(student_text or "").strip().lower()
    r_clean = str(reference_text or "").strip().lower()

    if not s_clean or not r_clean:
        return {
            "lex_exact_match": 0.0,
            "lex_char_levenshtein": 0.0,
            "lex_token_levenshtein": 0.0,
            "lex_token_jaccard": 0.0,
            "lex_token_precision": 0.0,
            "lex_token_recall": 0.0,
            "lex_token_f1": 0.0,
            "lex_bigram_f1": 0.0,
            "lex_trigram_f1": 0.0,
            "lex_rouge_l_f1": 0.0,
            "lex_rouge_l_precision": 0.0,
            "lex_rouge_l_recall": 0.0,
            "lex_length_ratio": 0.0
        }

    # 1. Exact match
    exact = 1.0 if s_clean == r_clean else 0.0

    # 2. Levenshtein normalized similarity
    char_lev = float(rapidfuzz.distance.Levenshtein.normalized_similarity(s_clean, r_clean))

    # Token-level Levenshtein
    s_toks = clean_tokens(s_clean)
    r_toks = clean_tokens(r_clean)
    token_lev = float(rapidfuzz.distance.Levenshtein.normalized_similarity(s_toks, r_toks)) if s_toks and r_toks else 0.0

    # 3. Token Jaccard & F1
    s_set = set(s_toks)
    r_set = set(r_toks)
    inter = len(s_set.intersection(r_set))
    union = len(s_set.union(r_set))
    jaccard = float(inter / union) if union > 0 else 0.0

    prec = float(inter / len(s_set)) if s_set else 0.0
    rec = float(inter / len(r_set)) if r_set else 0.0
    tok_f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # 4. N-gram overlaps
    def get_ngrams(tokens: List[str], n: int) -> Set[tuple]:
        if len(tokens) < n:
            return set()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def ngram_f1(s_t: List[str], r_t: List[str], n: int) -> float:
        s_ng = get_ngrams(s_t, n)
        r_ng = get_ngrams(r_t, n)
        if not s_ng or not r_ng:
            return 0.0
        i_cnt = len(s_ng.intersection(r_ng))
        p = i_cnt / len(s_ng)
        r = i_cnt / len(r_ng)
        return float(2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    bi_f1 = ngram_f1(s_toks, r_toks, 2)
    tri_f1 = ngram_f1(s_toks, r_toks, 3)

    # 5. True ROUGE-L via exact DP LCS
    r_prec, r_rec, r_f1 = compute_true_rouge_l(s_clean, r_clean)

    # 6. Length ratio
    len_ratio = float(min(len(s_clean), len(r_clean)) / max(len(s_clean), len(r_clean)))

    return {
        "lex_exact_match": exact,
        "lex_char_levenshtein": round(char_lev, 4),
        "lex_token_levenshtein": round(token_lev, 4),
        "lex_token_jaccard": round(jaccard, 4),
        "lex_token_precision": round(prec, 4),
        "lex_token_recall": round(rec, 4),
        "lex_token_f1": round(tok_f1, 4),
        "lex_bigram_f1": round(bi_f1, 4),
        "lex_trigram_f1": round(tri_f1, 4),
        "lex_rouge_l_f1": round(r_f1, 4),
        "lex_rouge_l_precision": round(r_prec, 4),
        "lex_rouge_l_recall": round(r_rec, 4),
        "lex_length_ratio": round(len_ratio, 4)
    }
