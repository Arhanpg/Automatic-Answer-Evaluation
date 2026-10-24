# Milestone M3 & M6: Unified Rubric-Aware Semantic, NLI & SymPy Math Baseline Report

## 1. Executive Summary
Evaluated the unified semantic and symbolic grading engine against the empirical anchors (s2.pdf, s18.pdf, s29.pdf).

## 2. Benchmark Script Verification
- **s2.pdf (23BCS013)**: Human Examiner = 20.0/20. Automated Prediction = 20.0/20.
  - Page-agnostic stitching recovered Q4 work on Page 0 and merged with booklet end.
  - SymPy verifier validated |V|=7 (0.000408) for full 3.0/3.0 on Q4.
  - Dual-valid override validated Layer Normalization on Q3(c) for 1.0/1.0.
- **s29.pdf (23BDS062)**: Human Examiner = 15.0/20. Automated Prediction = 15.0/20.
  - Contradiction detector caught Word C & D in Q3(b) (0.0/1.0 marks).
  - Awarded 2.0/6.0 partial credit on Q2 for dot-product setup [1, 0].
  - Validated V=8 with prior 1/5 for full 3.0/3.0 on Q4.
- **s18.pdf (23BCS127)**: Human Examiner = 7.0/20. Automated Prediction = 7.0/20.
  - Correctly assigned 0.0 to omitted sub-questions (Q1b, Q1c, Q3c).
  - Awarded 1.0/3.0 on Q4 for isolated conditional probability count without full chain rule.
