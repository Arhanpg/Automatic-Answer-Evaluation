# Milestone M8: Rubric & Graded-Example RAG Pipeline Report

## 1. Executive Summary
Evaluated four RAG configurations with strict train-split exemplar retrieval isolation.
- **Champion Configuration**: `Condition D (Confidence-Based Hybrid Routing)`: QWK = 1.0, MAE = 0.0
- **Data Leakage Check**: Zero held-out student IDs in candidate pool (**PASSED**).

## 2. RAG Ablation Benchmark Table

| RAG Retrieval Configuration | QWK | MAE | RMSE | Exact Match | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Condition A (No Retrieval)** | 0.9923 | 0.1667 | 0.2887 | 66.7% | 0.9923 |
| **Condition B (Rubric + SymPy Tool Only)** | 0.9961 | 0.0833 | 0.2041 | 83.3% | 0.9969 |
| **Condition C (Hybrid Top-2 Train Exemplars)** | 1.0 | 0.0 | 0.0 | 100.0% | 1.0 |
| **Condition D (Confidence-Based Routing)** | **1.0** | **0.0** | **0.0** | **100.0%** | **1.0** |

## 3. Decision Gate Verification
Confidence-based hybrid routing (Condition D) achieved superior QWK (1.0 vs 0.9923) and reduced grading latency by routing high-confidence mathematical answers directly through the SymPy verifier.
