"""
backend/feature_engineering.py -- 18-Dimensional Feature Vector Extraction for Automated Grading.

Constructs the standardized 18-feature representation for every (student_id, question_id) response:
[
    tfidf_sim, bm25_score, rouge_l, sbert_max_sim, sbert_mean_sim,
    criterion_coverage_ratio, contradiction_flag, math_sympy_score, key_term_recall,
    char_length, word_count, line_count, equation_density,
    ocr_mean_confidence, ocr_min_confidence, pass1_pass2_levenshtein,
    question_max_marks, is_math_question
]
"""
import re
import math
import numpy as np
from typing import Dict, List, Any, Optional
import rapidfuzz
from backend import semantic_scorer, math_verifier

MATH_CHARS = set("+=-\\/*^()[]{}|_0123456789<>.%")

def compute_rouge_l(student_text: str, model_text: str) -> float:
    """Computes ROUGE-L LCS F1 score."""
    s_words = semantic_scorer.clean_tokens(student_text)
    m_words = semantic_scorer.clean_tokens(model_text)
    if not s_words or not m_words:
        return 0.0

    # Longest Common Subsequence length
    matcher = rapidfuzz.distance.Levenshtein.distance(s_words, m_words)
    # Approximate LCS via Levenshtein
    lcs = max(0, (len(s_words) + len(m_words) - matcher) // 2)
    prec = lcs / len(s_words) if s_words else 0.0
    rec = lcs / len(m_words) if m_words else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))

def extract_features_for_question(
    student_id: str,
    question_id: str,
    student_text: str,
    model_answer: str,
    criteria: List[Dict[str, Any]],
    max_marks: float,
    ocr_meta: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Extracts the 18-dimensional feature vector for a single question response.
    """
    s_clean = str(student_text or "").strip()
    m_clean = str(model_answer or "").strip()
    ocr_meta = ocr_meta or {}

    is_math = 1.0 if question_id in ["2", "Q2", "4", "Q4"] else 0.0

    # 1. Lexical features: TF-IDF, BM25 proxy, ROUGE-L
    tfidf_sim = float(semantic_scorer.tfidf_cosine_similarity([s_clean], m_clean)[0]) if s_clean else 0.0
    lex = semantic_scorer.lexical_similarity(s_clean, m_clean)
    bm25_score = float(lex["token_jaccard"] * (1.0 + math.log(max(1, len(s_clean.split())))))
    rouge_l = compute_rouge_l(s_clean, m_clean)

    # 2. Semantic features: SBERT max and mean
    crit_sims = []
    for c in criteria:
        desc = c.get("description", "")
        sim = semantic_scorer.score_criterion_semantic(s_clean, desc, m_clean)
        crit_sims.append(sim)
    sbert_max_sim = float(max(crit_sims)) if crit_sims else tfidf_sim
    sbert_mean_sim = float(np.mean(crit_sims)) if crit_sims else tfidf_sim
    cov_ratio = float(sum(1 for s in crit_sims if s >= 0.4) / max(1, len(crit_sims)))

    # 3. Contradiction & SymPy Math Validity
    has_contra, _ = semantic_scorer.detect_contradictions(question_id, s_clean)
    contra_flag = 1.0 if has_contra else 0.0

    math_score = 0.0
    if is_math:
        if question_id in ["2", "Q2"]:
            # Check for attention components
            m_res, _ = math_verifier.verify_q2({"2a": s_clean, "2b": s_clean, "2c": s_clean})
            math_score = float(m_res / 6.0)
        elif question_id in ["4", "Q4"]:
            m_res, _ = math_verifier.verify_q4(s_clean)
            math_score = float(m_res / 3.0)
    else:
        math_score = sbert_mean_sim

    # 4. Domain key term recall
    m_tokens = set(semantic_scorer.clean_tokens(m_clean))
    s_tokens = set(semantic_scorer.clean_tokens(s_clean))
    key_term_recall = float(len(s_tokens.intersection(m_tokens)) / max(1, len(m_tokens)))

    # 5. Length & surface features
    char_length = float(len(s_clean))
    words = s_clean.split()
    word_count = float(len(words))
    line_count = float(max(1, s_clean.count("\n") + 1)) if s_clean else 0.0
    eq_chars = sum(1 for c in s_clean if c in MATH_CHARS)
    eq_density = float(eq_chars / max(1, len(s_clean)))

    # 6. OCR confidence & discrepancy features
    ocr_mean_conf = float(ocr_meta.get("mean_conf", 0.95))
    ocr_min_conf = float(ocr_meta.get("min_conf", 0.88))
    lev_discrepancy = float(ocr_meta.get("levenshtein_discrepancy", 0.0))

    feat_vector = np.array([
        tfidf_sim,                  # 0: tfidf_sim
        bm25_score,                 # 1: bm25_score
        rouge_l,                    # 2: rouge_l
        sbert_max_sim,              # 3: sbert_max_sim
        sbert_mean_sim,             # 4: sbert_mean_sim
        cov_ratio,                  # 5: criterion_coverage_ratio
        contra_flag,                # 6: contradiction_flag
        math_score,                 # 7: math_sympy_score
        key_term_recall,            # 8: key_term_recall
        char_length,                # 9: char_length
        word_count,                 # 10: word_count
        line_count,                 # 11: line_count
        eq_density,                 # 12: equation_density
        ocr_mean_conf,              # 13: ocr_mean_confidence
        ocr_min_conf,               # 14: ocr_min_confidence
        lev_discrepancy,            # 15: pass1_pass2_levenshtein
        float(max_marks),           # 16: question_max_marks
        is_math,                    # 17: is_math_question
    ], dtype=np.float32)

    return feat_vector

FEATURE_NAMES = [
    "tfidf_sim", "bm25_score", "rouge_l", "sbert_max_sim", "sbert_mean_sim",
    "criterion_coverage_ratio", "contradiction_flag", "math_sympy_score", "key_term_recall",
    "char_length", "word_count", "line_count", "equation_density",
    "ocr_mean_confidence", "ocr_min_confidence", "pass1_pass2_levenshtein",
    "question_max_marks", "is_math_question"
]
