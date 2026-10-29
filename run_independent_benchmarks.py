"""
run_independent_benchmarks.py - Independent Multi-Modal OCR, VLM & 27-Model Benchmark Runner.

Executes the 4-stage preprocessing, PaddlePaddle OCR, Tesseract OCR, and Vision-Language Model
independent benchmarks across all 27 models with strict split isolation (zero data leakage)
and saves all outputs to outputs/independent_benchmarks/ without modifying existing results.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.independent_pipeline import run_independent_benchmark_suite

if __name__ == "__main__":
    run_independent_benchmark_suite()

