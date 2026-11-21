# Phase 8 Dataset Split Integrity and Historical Exposure Report

## 1. Executive Summary
- **Total Booklets in Corpus:** 37
- **Development Pool Size:** 31 booklets
- **Historical Test Pool Size:** 6 booklets (`s7`, `s14`, `s24`, `s29`, `s30`, `s37`)
- **Methodological Exposure Status:** `PREVIOUSLY_EXPOSED_HISTORICAL_SPLIT`
- **Partition Disjointness:** VERIFIED (0 student overlap across all folds and test sets)

---

## 2. Historical Exposure Finding (Forensic Audit Defect F)
During Phase 1–7 exploratory experiments, student scripts `s7, s14, s24, s29, s30, s37` were repeatedly inspected, tuned upon, and evaluated across multiple scripts:
- `scripts/run_clean_research_pipeline.py`
- `scripts/run_genuine_experiments.py`
- `research_final/03_TRANSCRIPTION_ADJUDICATION.csv`

Consequently, reporting generalization metrics on these 6 students as "unseen held-out test data" constitutes methodological over-optimism. For Phase 8 and subsequent publications:
1. These 6 booklets are formally cataloged as `previously_exposed_historical_split`.
2. Primary cross-validation benchmarks must be evaluated using nested 5-fold cross-validation on the 31-student development pool.
3. Any evaluation on the 6 exposed booklets must be explicitly reported as a historical ablation benchmark, not pure zero-shot generalization.

---

## 3. Recommended 5-Fold Cross-Validation Partition
| Fold Index | Booklet Count | Student Identifiers |
|---|---|---|
| **Fold 1** | 7 | s1, s15, s2, s25, s31, s36, s9 |
| **Fold 2** | 6 | s10, s16, s20, s26, s32, s4 |
| **Fold 3** | 6 | s11, s17, s21, s27, s33, s5 |
| **Fold 4** | 6 | s12, s18, s22, s28, s34, s6 |
| **Fold 5** | 6 | s13, s19, s23, s3, s35, s8 |
| **Historical Exposed Split** | 6 | s14, s24, s29, s30, s37, s7 |

All splits are strictly booklet-disjoint. Zero pages, answer regions, or rubrics from the same physical examination booklet appear in more than one partition.
