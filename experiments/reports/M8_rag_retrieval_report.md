# Milestone M8: Rubric and Graded-Example RAG

## Benchmark conditions

The RAG artifact compares four configurations over 12 generated/benchmark
instances with a 25-script train pool:

| Condition | MAE | RMSE | Exact match | QWK |
|---|---:|---:|---:|---:|
| No retrieval | 0.2500 | 0.5401 | 75.0% | 0.9732 |
| Rubric + SymPy | 0.2083 | 0.3227 | 58.3% | 0.9901 |
| Hybrid top-2 train examples | 0.0417 | 0.1443 | 91.7% | 0.9981 |
| Confidence-based routing | 0.0000 | 0.0000 | 100.0% | 1.0000 |

The train-pool isolation assertion is recorded as passed. The perfect final
row is a generated benchmark result and should not be interpreted as universal
grading accuracy.
