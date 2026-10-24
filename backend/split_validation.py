"""
backend/split_validation.py -- Student-level split partitioning and leakage protection.

Enforces complete student-script level isolation across all 37 examination booklets:
- 25 train booklets (100 question responses)
- 6 validation booklets (24 question responses)
- 6 locked test booklets (24 question responses)
Stratifies anchor scripts (s2 at 20/20, s29 at 15/20, s18 at 7/20) across val and test.
Provides 5-Fold GroupKFold splits and SHA-256 split hash verification.
"""
import os
import json
import hashlib
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import GroupKFold
from backend.output_manager import DedicatedOutputManager

# Definitive anchor marks for stratification
ANCHOR_MARKS = {
    "s2": 20.0,
    "s29": 15.0,
    "s18": 7.0,
}

DEFAULT_DATASET_DIR = r"F:\Automatic answer evaluation dataset"

def get_all_student_ids(dataset_dir: str = DEFAULT_DATASET_DIR) -> List[str]:
    """Discovers all 37 student booklet IDs."""
    all_pdfs_dir = os.path.join(dataset_dir, "All PDF's")
    if os.path.exists(all_pdfs_dir):
        files = [f for f in os.listdir(all_pdfs_dir) if f.startswith("s") and f.endswith(".pdf")]
        ids = sorted([os.path.splitext(f)[0] for f in files], key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
        if len(ids) == 37:
            return ids
    # Fallback to standard range s1..s37
    return [f"s{i}" for i in range(1, 38)]

def build_student_splits(student_ids: List[str]) -> Dict[str, Any]:
    """Partitions the 37 booklets into 25 train, 6 val, and 6 test scripts with anchor stratification."""
    # Anchors:
    # High: s2 (20/20)
    # Med: s29 (15/20)
    # Low: s18 (7/20)
    
    # Place s2 and s18 in val, s29 in test (or stratified pairs)
    # Val: 6 scripts -> s2 (high), s18 (low), s5, s12, s21, s33
    val_set = {"s2", "s18", "s5", "s12", "s21", "s33"}
    # Test: 6 scripts -> s29 (med), s7, s14, s24, s30, s37
    test_set = {"s29", "s7", "s14", "s24", "s30", "s37"}
    
    # Remaining 25 scripts are train
    train_set = [s for s in student_ids if s not in val_set and s not in test_set]
    val_list = [s for s in student_ids if s in val_set]
    test_list = [s for s in student_ids if s in test_set]

    assert_no_leakage(train_set, val_list, test_list)
    assert len(train_set) == 25, f"Expected 25 train scripts, got {len(train_set)}"
    assert len(val_list) == 6, f"Expected 6 val scripts, got {len(val_list)}"
    assert len(test_list) == 6, f"Expected 6 test scripts, got {len(test_list)}"

    # Generate 5-fold cross validation folds for the 31 non-test scripts
    cv_students = train_set + val_list
    gkf = GroupKFold(n_splits=5)
    # Each student has 4 questions
    cv_records = []
    for s_id in cv_students:
        for q_id in ["Q1", "Q2", "Q3", "Q4"]:
            cv_records.append({"student_id": s_id, "question_id": q_id})
    
    groups = [r["student_id"] for r in cv_records]
    cv_folds = []
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(cv_records, groups=groups)):
        tr_studs = sorted(list({cv_records[i]["student_id"] for i in tr_idx}))
        vl_studs = sorted(list({cv_records[i]["student_id"] for i in val_idx}))
        cv_folds.append({
            "fold": fold_idx + 1,
            "train_students": tr_studs,
            "val_students": vl_studs,
        })

    split_manifest = {
        "dataset_name": "IIIT_Dharwad_DS317_Midsem",
        "total_booklets": len(student_ids),
        "total_responses": len(student_ids) * 4,
        "splits": {
            "train": {
                "count": len(train_set),
                "student_ids": train_set,
                "response_count": len(train_set) * 4,
            },
            "validation": {
                "count": len(val_list),
                "student_ids": val_list,
                "response_count": len(val_list) * 4,
                "anchors": ["s2 (20/20)", "s18 (7/20)"],
            },
            "test": {
                "count": len(test_list),
                "student_ids": test_list,
                "response_count": len(test_list) * 4,
                "anchors": ["s29 (15/20)"],
            },
        },
        "grouped_5fold_cv": cv_folds,
    }

    manifest_json = json.dumps(split_manifest, sort_keys=True)
    split_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    split_manifest["split_sha256"] = split_hash

    return split_manifest

def assert_no_leakage(train_ids: List[str], val_ids: List[str], test_ids: List[str]):
    """Strict runtime assertion guarding against inter-split data leakage."""
    s_tr, s_val, s_tst = set(train_ids), set(val_ids), set(test_ids)
    tr_val = s_tr.intersection(s_val)
    tr_tst = s_tr.intersection(s_tst)
    val_tst = s_val.intersection(s_tst)
    if tr_val:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: Train and Val overlap on: {tr_val}")
    if tr_tst:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: Train and Test overlap on: {tr_tst}")
    if val_tst:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: Val and Test overlap on: {val_tst}")

def initialize_and_save_splits(output_mgr: DedicatedOutputManager, dataset_dir: str = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    """Computes, validates, and persists student split manifest and M1 report."""
    student_ids = get_all_student_ids(dataset_dir)
    splits = build_student_splits(student_ids)
    output_mgr.save_dataset_audit(splits)

    report_content = f"""# Milestone M1: Dataset Schema & Student Split Validation Audit

## 1. Executive Summary
- **Dataset**: IIIT Dharwad 7th Sem B.Tech LLMs Mid-Semester Examination (DS317 / DS342)
- **Total Answer Booklets**: {splits['total_booklets']} scripts (s1 to s37)
- **Total Responses**: {splits['total_responses']} question responses
- **Split SHA-256 Checksum**: `{splits['split_sha256']}`

## 2. Student-Level Partitioning
To guarantee zero penmanship and examiner-bias leakage, data is partitioned strictly at the student booklet level:
- **Train Split (70%)**: 25 Booklets (100 responses)
  `{', '.join(splits['splits']['train']['student_ids'])}`
- **Validation Split (15%)**: 6 Booklets (24 responses)
  `{', '.join(splits['splits']['validation']['student_ids'])}`
  *Anchors included*: `s2.pdf` (Score 20/20, High Tier), `s18.pdf` (Score 7/20, Low Tier)
- **Locked Test Split (15%)**: 6 Booklets (24 responses)
  `{', '.join(splits['splits']['test']['student_ids'])}`
  *Anchors included*: `s29.pdf` (Score 15/20, Medium Tier)

## 3. Grouped Cross-Validation Protocol
5-Fold Grouped Cross Validation (`GroupKFold(n_splits=5)`) is configured across all 31 non-test scripts. No student script is ever split across training and validation folds within any iteration.

## 4. Leakage Verification Gate
- Inter-partition intersection: **0 scripts (PASS)**
- All question parts of a booklet remain strictly co-located on the same partition.
"""
    output_mgr.save_report("M1_dataset_and_rubric_audit.md", report_content)
    return splits

if __name__ == "__main__":
    mgr = DedicatedOutputManager()
    manifest = initialize_and_save_splits(mgr)
    print(f"Dataset split initialized successfully. SHA256: {manifest['split_sha256']}")
