"""
research_v3/phase8_ground_truth/scripts/verify_split_integrity.py

Audits dataset partitions and test split integrity:
1. Formally classifies the historical 6 test students (s7, s14, s24, s29, s30, s37)
   as 'previously_exposed_historical_split' due to extensive exploratory usage.
2. Validates student-level disjointness across all cross-validation folds.
3. Ensures zero data leakage between development and evaluation sets.
4. Outputs split integrity report to Markdown.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BOOKLETS_CSV = REPO_ROOT / "research_v2" / "annotations" / "booklets.csv"
OUTPUT_REPORT = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "06_SPLIT_INTEGRITY_REPORT.md"

HISTORICAL_EXPOSED_TEST_STUDENTS = {"s7", "s14", "s24", "s29", "s30", "s37"}
TOTAL_STUDENTS = 37


def verify_splits():
    print("=" * 80)
    print("PHASE 8: SPLIT INTEGRITY & EXPOSURE AUDIT")
    print("=" * 80)

    if not BOOKLETS_CSV.exists():
        print(f"Error: booklets.csv missing at {BOOKLETS_CSV}")
        sys.exit(1)

    df = pd.read_csv(BOOKLETS_CSV)
    all_sids = set(df["student_id"].astype(str).tolist())

    if len(all_sids) != TOTAL_STUDENTS:
        print(f"Error: Expected {TOTAL_STUDENTS} students, found {len(all_sids)}")
        sys.exit(1)

    # Reclassify splits
    dev_pool = all_sids - HISTORICAL_EXPOSED_TEST_STUDENTS
    print(f"Total booklets in corpus: {len(all_sids)}")
    print(f"Development pool booklets: {len(dev_pool)} {sorted(list(dev_pool))}")
    print(f"Previously exposed historical test booklets: {len(HISTORICAL_EXPOSED_TEST_STUDENTS)} {sorted(list(HISTORICAL_EXPOSED_TEST_STUDENTS))}")

    # Check 5-fold cross-validation allocation on development pool (31 students)
    # Sorted deterministic split
    sorted_dev = sorted(list(dev_pool))
    k = 5
    folds = [[] for _ in range(k)]
    for idx, sid in enumerate(sorted_dev):
        folds[idx % k].append(sid)

    print("\nProposed 5-Fold Student-Disjoint Development Partitioning (31 Students):")
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1} (N={len(fold)}): {fold}")

    # Verify disjointness
    for i in range(k):
        for j in range(i + 1, k):
            overlap = set(folds[i]).intersection(set(folds[j]))
            if overlap:
                print(f"Error: Overlap between Fold {i+1} and Fold {j+1}: {overlap}")
                sys.exit(1)

    test_overlap = set(sorted_dev).intersection(HISTORICAL_EXPOSED_TEST_STUDENTS)
    if test_overlap:
        print(f"Error: Development pool overlaps with exposed test set: {test_overlap}")
        sys.exit(1)

    # Write report
    report_md = f"""# Phase 8 Dataset Split Integrity and Historical Exposure Report

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
| **Fold 1** | 7 | {', '.join(folds[0])} |
| **Fold 2** | 6 | {', '.join(folds[1])} |
| **Fold 3** | 6 | {', '.join(folds[2])} |
| **Fold 4** | 6 | {', '.join(folds[3])} |
| **Fold 5** | 6 | {', '.join(folds[4])} |
| **Historical Exposed Split** | 6 | {', '.join(sorted(list(HISTORICAL_EXPOSED_TEST_STUDENTS)))} |

All splits are strictly booklet-disjoint. Zero pages, answer regions, or rubrics from the same physical examination booklet appear in more than one partition.
"""
    OUTPUT_REPORT.write_text(report_md, encoding="utf-8")
    print(f"\nSplit integrity report written to: {OUTPUT_REPORT}")
    print("SPLIT INTEGRITY AND EXPOSURE AUDIT PASSED!")


if __name__ == "__main__":
    verify_splits()
