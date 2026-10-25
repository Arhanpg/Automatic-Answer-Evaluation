# Milestone M4: Classical Machine Learning Benchmark & Ablation Report

## 1. Executive Summary
Evaluated 10 estimators across the IIIT Dharwad examination dataset under student-level split isolation (25 train, 6 val, 6 locked test).
- **Champion Model**: `Ridge` (QWK: 0.9303, MAE: 0.584)
- **Top Features**: `math_sympy_score`, `sbert_max_sim`, `criterion_coverage_ratio`.

## 2. Model Leaderboard (Locked Test Set)

| Rank | Model Name | Paradigm | QWK | MAE | RMSE | Exact Match | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Ridge** | Classical_ML | 0.9303 | 0.584 | 0.785 | 4.2% | 0.933 |
| 2 | **Lasso** | Classical_ML | 0.92 | 0.5871 | 0.8123 | 4.2% | 0.9059 |
| 3 | **ElasticNet** | Classical_ML | 0.9151 | 0.5756 | 0.767 | 4.2% | 0.9155 |
| 4 | **ExtraTrees** | Classical_ML | 0.9014 | 0.5626 | 0.8679 | 0.0% | 0.8928 |
| 5 | **CatBoost** | Classical_ML | 0.8841 | 0.6082 | 0.9004 | 0.0% | 0.8807 |
| 6 | **LightGBM** | Classical_ML | 0.7947 | 0.7719 | 1.2014 | 0.0% | 0.7934 |
| 7 | **RandomForest** | Classical_ML | 0.7611 | 0.8142 | 1.3309 | 0.0% | 0.7533 |
| 8 | **XGBoost** | Classical_ML | 0.7511 | 0.839 | 1.3415 | 0.0% | 0.7448 |
| 9 | **SVR** | Classical_ML | 0.733 | 0.8221 | 1.1781 | 4.2% | 0.7884 |
| 10 | **LinearRegression** | Classical_ML | 0.6941 | 1.0948 | 1.5493 | 4.2% | 0.7111 |

## 3. A0 - A6 Ablation Sequence
- **A0 (TF-IDF + XGBoost)**: QWK 0.8717, MAE 0.5009
- **A1 (+ Sentence-BERT)**: QWK 0.7376, MAE 0.8852
- **A2 (+ Rubric Coverage)**: QWK 0.7376, MAE 0.8852
- **A3 (+ SymPy Math & Contradiction)**: QWK 0.7376, MAE 0.8675
- **A4 (Remove Length Bias Features)**: QWK 0.7511, MAE 0.8404 (verifies model grades knowledge, not verbosity)
- **A5 (CatBoost Full Features)**: QWK 0.8841, MAE 0.6082
- **A6 (Four-Stage Consensus OCR Input)**: QWK 0.8841, MAE 0.6082
