"""
research_v3/phase9_ocr_benchmarking/tests/test_ocr_benchmark.py

Unit tests for Phase 9 isolated OCR benchmarking and preprocessing pipeline:
1. Metric validation: CER, WER, Exact Match, Math F1, Numeric F1
2. Preprocessing pipeline transformations (Otsu, Sauvola, CLAHE, Illumination)
3. Mathematical symbol and numeric token extractors
4. Omission evaluation handling
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np
from PIL import Image

from research_v3.phase9_ocr_benchmarking.metrics import (
    compute_cer, compute_wer, compute_exact_match,
    extract_math_tokens, extract_numbers,
    compute_math_metrics, compute_numeric_metrics,
    evaluate_prediction
)
from research_v3.phase9_ocr_benchmarking.preprocessing import (
    preprocess_raw, preprocess_grayscale, preprocess_otsu,
    preprocess_sauvola, preprocess_clahe,
    preprocess_illumination_correction, preprocess_composite
)


def test_cer_calculation():
    # Exact match
    assert compute_cer("hello", "hello") == 0.0
    # Single substitution: 1 / 5 = 0.2
    assert abs(compute_cer("hello", "hallo") - 0.2) < 1e-4
    # Insertion exceeding length: can exceed 1.0
    assert compute_cer("a", "aaaa") == 3.0
    # Empty reference and empty hypothesis
    assert compute_cer("", "") == 0.0


def test_wer_calculation():
    assert compute_wer("the quick brown fox", "the quick brown fox") == 0.0
    # 1 word change in 4 words: 0.25
    assert abs(compute_wer("the quick brown fox", "the fast brown fox") - 0.25) < 1e-4
    # Empty
    assert compute_wer("", "") == 0.0


def test_math_token_extraction():
    sample = "P(ML|like) = (0+1)/(2+7) = 1/9, Softmax([1, 0]) = [0.73, 0.27]"
    tokens = extract_math_tokens(sample)
    assert "=" in tokens
    assert "/" in tokens
    assert "+" in tokens
    assert "softmax" in tokens
    assert "p(" in tokens


def test_numeric_token_extraction():
    sample = "Q.K1 = 1, Q.K2 = 0, Output vector = [1.46, 0.81], Laplace smoothing with V=8"
    nums = extract_numbers(sample)
    assert "1" in nums
    assert "0" in nums
    assert "1.46" in nums
    assert "0.81" in nums
    assert "8" in nums


def test_math_metrics():
    ref = "Softmax([1, 0]) = [0.73, 0.27]"
    hyp = "Softmax([1, 0]) = [0.73, 0.27]"
    res = compute_math_metrics(ref, hyp)
    assert res["math_f1"] == 1.0

    # Dropped symbols
    hyp_partial = "Softmax [1, 0]"
    res_partial = compute_math_metrics(ref, hyp_partial)
    assert res_partial["math_recall"] < 1.0


def test_omission_evaluation():
    # Correctly predicted blank on omitted slot
    res_correct = evaluate_prediction("", "", is_omitted=True)
    assert res_correct["cer"] == 0.0
    assert res_correct["wer"] == 0.0
    assert res_correct["exact_match"] == 1.0
    assert res_correct["omission_correct"] == 1.0

    # Hallucinated text on omitted slot
    res_hallucinated = evaluate_prediction("", "Some text", is_omitted=True)
    assert res_hallucinated["cer"] == 1.0
    assert res_hallucinated["exact_match"] == 0.0
    assert res_hallucinated["omission_correct"] == 0.0


def test_preprocessing_transforms():
    # Create synthetic RGB gradient image
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :] = [200, 210, 220]
    # Add fake pen stroke
    arr[40:60, 40:60] = [20, 20, 30]
    img = Image.fromarray(arr)

    # Test all transforms produce valid non-empty images
    for fn in [
        preprocess_raw, preprocess_grayscale, preprocess_otsu,
        preprocess_sauvola, preprocess_clahe,
        preprocess_illumination_correction, preprocess_composite
    ]:
        out = fn(img)
        assert isinstance(out, Image.Image)
        assert out.size == (100, 100)
