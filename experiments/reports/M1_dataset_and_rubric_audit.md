# Milestone M1: Dataset Schema & Student Split Validation Audit

## 1. Executive Summary
- **Dataset**: IIIT Dharwad 7th Sem B.Tech LLMs Mid-Semester Examination (DS317 / DS342)
- **Total Answer Booklets**: 37 scripts (s1 to s37)
- **Total Responses**: 148 question responses
- **Split SHA-256 Checksum**: `41327d9f77e8a40dcc6c027a9250eed79175cc4b6fa2cbdd6f4cc479f40c06e1`

## 2. Student-Level Partitioning
To guarantee zero penmanship and examiner-bias leakage, data is partitioned strictly at the student booklet level:
- **Train Split (70%)**: 25 Booklets (100 responses)
  `s1, s3, s4, s6, s8, s9, s10, s11, s13, s15, s16, s17, s19, s20, s22, s23, s25, s26, s27, s28, s31, s32, s34, s35, s36`
- **Validation Split (15%)**: 6 Booklets (24 responses)
  `s2, s5, s12, s18, s21, s33`
  *Anchors included*: `s2.pdf` (Score 20/20, High Tier), `s18.pdf` (Score 7/20, Low Tier)
- **Locked Test Split (15%)**: 6 Booklets (24 responses)
  `s7, s14, s24, s29, s30, s37`
  *Anchors included*: `s29.pdf` (Score 15/20, Medium Tier)

## 3. Grouped Cross-Validation Protocol
5-Fold Grouped Cross Validation (`GroupKFold(n_splits=5)`) is configured across all 31 non-test scripts. No student script is ever split across training and validation folds within any iteration.

## 4. Leakage Verification Gate
- Inter-partition intersection: **0 scripts (PASS)**
- All question parts of a booklet remain strictly co-located on the same partition.
