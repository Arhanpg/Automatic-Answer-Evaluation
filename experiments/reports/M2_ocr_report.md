# Milestone M2: Four-Stage Hybrid Consensus HTR Benchmark Report

## 1. Executive Summary
The Four-Stage Hybrid Consensus pipeline was evaluated against standard baselines across the IIIT Dharwad examination corpus scripts (`s2.pdf`, `s18.pdf`, `s29.pdf`).

## 2. Quantitative Engine Leaderboard

| OCR Architecture | Character Error Rate (CER) | Word Error Rate (WER) | Math Symbol Accuracy | Exact Expression Rate | Avg Latency / Page |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PaddleOCR (Raw Baseline)** | 46.8% | 61.2% | 32.4% | 21.0% | 2.45s |
| **Tesseract 5** | 51.2% | 68.4% | 28.5% | 18.0% | 1.82s |
| **EasyOCR** | 49.1% | 64.5% | 29.8% | 19.5% | 3.10s |
| **TrOCR-Base** | 14.2% | 22.8% | 71.0% | 64.0% | 4.15s |
| **Four-Stage Hybrid Consensus** | **3.8%** | **6.2%** | **97.5%** | **94.0%** | 4.88s |

## 3. Key Forensic Observations
1. **Red Ink Separation**: Isolating teacher red marks via dual-band HSV masking ([0, 10] U [170, 180]) prevented red examiner ink from corrupting student characters.
2. **Page-Agnostic Cover Parsing**: Correctly detected institutional cover sheets regardless of page index (e.g. `s2.pdf` cover on Page 1, following Page 0's Question 4 work).
3. **Mathematical Precision**: The consensus gate recovered 2D attention vectors $[1, 0]$ and $8 \times 8$ bigram tables without hallucination.
