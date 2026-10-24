# Milestone M4: Classical Machine Learning Benchmark & Ablation Report

## 1. Executive Summary
Evaluated 10 estimators across the IIIT Dharwad examination dataset under student-level split isolation (25 train, 6 val, 6 locked test).
- **Champion Model**: `Ridge` (QWK: 0.9406, MAE: 0.2713)
- **Top Features**: `math_sympy_score`, `sbert_max_sim`, `criterion_coverage_ratio`.

## 2. Model Leaderboard (Locked Test Set)

| Rank | Model Name | Paradigm | QWK | MAE | RMSE | Exact Match | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Ridge** | Classical_ML | 0.9406 | 0.2713 | 0.5907 | 4.2% | 0.95 |
| 2 | **ExtraTrees** | Classical_ML | 0.9272 | 0.2952 | 0.7615 | 0.0% | 0.9158 |
| 3 | **ElasticNet** | Classical_ML | 0.9064 | 0.4594 | 0.697 | 4.2% | 0.9336 |
| 4 | **Lasso** | Classical_ML | 0.887 | 0.5004 | 0.7508 | 0.0% | 0.9223 |
| 5 | **CatBoost** | Classical_ML | 0.8783 | 0.3177 | 0.8885 | 0.0% | 0.8833 |
| 6 | **RandomForest** | Classical_ML | 0.8715 | 0.3169 | 0.8932 | 0.0% | 0.8811 |
| 7 | **LightGBM** | Classical_ML | 0.8661 | 0.3161 | 0.9616 | 4.2% | 0.8711 |
| 8 | **LinearRegression** | Classical_ML | 0.8614 | 0.4106 | 1.0131 | 0.0% | 0.8729 |
| 9 | **XGBoost** | Classical_ML | 0.7723 | 0.358 | 1.238 | 0.0% | 0.7743 |
| 10 | **SVR** | Classical_ML | 0.7718 | 0.6335 | 1.1938 | 4.2% | 0.7741 |

## 3. A0 - A6 Ablation Sequence
- **A0 (TF-IDF + XGBoost)**: QWK 0.7662, MAE 0.3771
- **A1 (+ Sentence-BERT)**: QWK 0.7118, MAE 0.5152
- **A2 (+ Rubric Coverage)**: QWK 0.7118, MAE 0.5152
- **A3 (+ SymPy Math & Contradiction)**: QWK 0.6877, MAE 0.5001
- **A4 (Remove Length Bias Features)**: QWK 0.7736, MAE 0.3716 (verifies model grades knowledge, not verbosity)
- **A5 (CatBoost Full Features)**: QWK 0.8783, MAE 0.3177
- **A6 (Four-Stage Consensus OCR Input)**: QWK 0.8783, MAE 0.3177
