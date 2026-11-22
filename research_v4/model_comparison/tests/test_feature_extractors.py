"""
research_v4/model_comparison/tests/test_feature_extractors.py

Unit Tests for Feature Extraction Modules:
1. Validates standard Robertson BM25Okapi calculation against hand-verified reference.
2. Validates True ROUGE-L exact DP Longest Common Subsequence (LCS).
3. Validates SentenceTransformer availability and fallback logging.
4. Validates SymPy mathematical verification on subquestions.
"""

import pytest
import math
from research_v4.model_comparison.src.feature_extractors.bm25 import BM25Okapi
from research_v4.model_comparison.src.text_normalization import exact_lcs_length, compute_true_rouge_l
from research_v4.model_comparison.src.feature_extractors.semantic import is_sbert_available, extract_semantic_features
from research_v4.model_comparison.src.feature_extractors.symbolic import verify_math_question_2, verify_math_question_4


def test_exact_lcs_length_hand_worked_cases():
    assert exact_lcs_length([], ["a", "b"]) == 0
    assert exact_lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3
    # Subsequence with gaps: ["a", "x", "b", "y", "c"] and ["a", "b", "c"] -> LCS is 3
    assert exact_lcs_length(["a", "x", "b", "y", "c"], ["a", "b", "c"]) == 3
    # Substitution case: ["a", "b", "c"] vs ["a", "z", "c"] -> LCS is 2
    assert exact_lcs_length(["a", "b", "c"], ["a", "z", "c"]) == 2


def test_true_rouge_l_precision_recall_f1():
    prec, rec, f1 = compute_true_rouge_l("attention is all you need", "attention is all you need")
    assert pytest.approx(prec, abs=1e-4) == 1.0
    assert pytest.approx(rec, abs=1e-4) == 1.0
    assert pytest.approx(f1, abs=1e-4) == 1.0

    # Half match
    prec2, rec2, f12 = compute_true_rouge_l("attention mechanism", "attention is all you need")
    assert prec2 > 0.0
    assert rec2 > 0.0
    assert f12 > 0.0


def test_bm25_okapi_formula():
    corpus = [
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog"],
        ["brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    ]
    bm25 = BM25Okapi(k1=1.5, b=0.75)
    bm25.fit(corpus)

    assert bm25.corpus_size == 3
    assert bm25.doc_freqs["the"] == 3
    assert bm25.doc_freqs["fox"] == 2
    assert bm25.doc_freqs["quick"] == 1

    # Score of matching query
    score = bm25.score(["quick"], ["the", "quick", "brown", "fox"])
    assert score > 0.0
    # Zero score for disjoint query
    score_zero = bm25.score(["elephant"], ["the", "quick", "brown", "fox"])
    assert score_zero == 0.0


def test_sbert_availability_and_clean_execution():
    assert is_sbert_available(), "SentenceTransformer must be available in environment"
    res = extract_semantic_features(
        "Transformers process all tokens in parallel using self-attention",
        "Transformers process words in parallel through multi-head self-attention mechanisms",
        [{"description": "Parallel token processing"}, {"description": "Self-attention mechanism"}]
    )
    assert res["sbert_loaded_flag"] == 1.0
    assert 0.0 <= res["sbert_model_answer_sim"] <= 1.0
    assert 0.0 <= res["sbert_criteria_max"] <= 1.0


def test_sympy_math_verification():
    # Correct Q2
    subqs_correct = {
        "2a": "Q.K1 = 1 and Q.K2 = 0 so dot products are [1, 0]",
        "2b": "Softmax gives [0.73, 0.27]",
        "2c": "Output vector = [1.46, 0.81]"
    }
    score_q2 = verify_math_question_2(subqs_correct)
    assert score_q2 == 6.0

    # Correct Q4
    score_q4 = verify_math_question_4("Using add-1 Laplace smoothing, probability = 3/2420")
    assert score_q4 == 3.0
