# Milestone M4: Classical Machine Learning Benchmark & Ablation Report

## 1. Executive Summary
Evaluated 10 estimators across the IIIT Dharwad examination dataset under student-level split isolation (25 train, 6 val, 6 locked test).
- **Champion Model**: `Ridge` (QWK: 0.7591, MAE: 1.4728)
- **Top Features**: `math_sympy_score`, `sbert_max_sim`, `criterion_coverage_ratio`.

## 2. Model Leaderboard (Locked Test Set)

| Rank | Model Name | Paradigm | QWK | MAE | RMSE | Exact Match | Pearson r |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Ridge** | Classical_ML | 0.7591 | 1.4728 | 1.7237 | 12.5% | 0.8441 |
| 2 | **LightGBM** | Classical_ML | 0.7561 | 1.4864 | 1.7506 | 4.2% | 0.8438 |
| 3 | **RandomForest** | Classical_ML | 0.7389 | 1.4615 | 1.816 | 12.5% | 0.8344 |
| 4 | **ElasticNet** | Classical_ML | 0.7218 | 1.5477 | 1.8721 | 8.3% | 0.7974 |
| 5 | **Lasso** | Classical_ML | 0.6888 | 1.5814 | 1.9243 | 4.2% | 0.7798 |
| 6 | **CatBoost** | Classical_ML | 0.6229 | 1.6046 | 2.1397 | 12.5% | 0.7279 |
| 7 | **XGBoost** | Classical_ML | 0.6045 | 1.7 | 2.2599 | 12.5% | 0.7152 |
| 8 | **ExtraTrees** | Classical_ML | 0.5982 | 1.6448 | 2.2357 | 4.2% | 0.6964 |
| 9 | **SVR** | Classical_ML | 0.5911 | 1.486 | 2.2199 | 29.2% | 0.6637 |
| 10 | **LinearRegression** | Classical_ML | 0.5687 | 1.6885 | 2.4777 | 25.0% | 0.5917 |

## 3. A0 - A6 Ablation Sequence
- **A0 (TF-IDF + XGBoost)**: QWK 0.6088, MAE 1.7909
- **A1 (+ Sentence-BERT)**: QWK 0.6045, MAE 1.7086
- **A2 (+ Rubric Coverage)**: QWK 0.5999, MAE 1.7175
- **A3 (+ SymPy Math & Contradiction)**: QWK 0.6072, MAE 1.7204
- **A4 (Remove Length Bias Features)**: QWK 0.6045, MAE 1.7026 (verifies model grades knowledge, not verbosity)
- **A5 (CatBoost Full Features)**: QWK 0.6229, MAE 1.6046
- **A6 (Four-Stage Consensus OCR Input)**: QWK 0.6229, MAE 1.6046
