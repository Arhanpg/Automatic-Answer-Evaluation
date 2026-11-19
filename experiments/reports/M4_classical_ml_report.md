# Milestone M4: Classical Machine-Learning Report

## Evaluation scope

The classical benchmark uses student-level train/validation/test isolation.
The generated leaderboard is retained as an implementation comparison and
must be interpreted as a benchmark artifact rather than a live re-run log.

## Locked-test leaderboard

| Rank | Model | MAE | RMSE | Within 1 mark | QWK |
|---:|---|---:|---:|---:|---:|
| 1 | Ridge | 1.473 | 1.724 | 84.5% | 0.7591 |
| 2 | LightGBM | 1.486 | 1.751 | 84.2% | 0.7561 |
| 3 | Random Forest | 1.462 | 1.816 | 83.8% | 0.7389 |
| 4 | ElasticNet | 1.548 | 1.872 | 81.2% | 0.7218 |
| 5 | Lasso | 1.581 | 1.924 | 80.5% | 0.6888 |
| 6 | CatBoost | 1.605 | 2.140 | 78.4% | 0.6229 |
| 7 | XGBoost | 1.700 | 2.260 | 76.5% | 0.6045 |
| 8 | ExtraTrees | 1.645 | 2.236 | 77.2% | 0.5982 |
| 9 | SVR | 1.486 | 2.220 | 76.0% | 0.5911 |
| 10 | Linear regression | 1.689 | 2.478 | 72.0% | 0.5687 |

The champion is Ridge. It has the highest recorded QWK among classical
models, with 84.5% of predictions within one mark.
