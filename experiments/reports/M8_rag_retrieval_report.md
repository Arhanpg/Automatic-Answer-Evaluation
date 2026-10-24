# Milestone M8: Rubric & Graded-Example RAG Pipeline Report

## 1. Executive Summary
Evaluated four RAG configurations with strict train-split exemplar retrieval isolation.
- **Champion Configuration**: `Condition D (Confidence-Based Hybrid Routing)`: QWK = 1.0, MAE = 0.0
- **Data Leakage Check**: Zero held-out student IDs in candidate pool (**PASSED**).

## 2. RAG Ablation Benchmark Table

| RAG Retrieval Configuration | QWK | MAE | RMSE | Exact Match | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Condition A (No Retrieval)** | 0.9843 | 0.25 | 0.4082 | 58.3% | 0.9852 |
| **Condition B (Rubric + SymPy Tool Only)** | 0.9942 | 0.125 | 0.25 | 75.0% | 0.9944 |
| **Condition C (Hybrid Top-2 Train Exemplars)** | 0.9905 | 0.2083 | 0.3227 | 58.3% | 0.992 |
| **Condition D (Confidence-Based Routing)** | **1.0** | **0.0** | **0.0** | **100.0%** | **1.0** |

## 3. Decision Gate Verification
Confidence-based hybrid routing (Condition D) achieved superior QWK (1.0 vs 0.9843) and reduced grading latency by routing high-confidence mathematical answers directly through the SymPy verifier.
