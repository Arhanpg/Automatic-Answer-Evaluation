"""
research_v4/model_comparison/src/splits.py

Grouped Cross-Validation and Exposure Audit Partitioning:
1. 5-Fold GroupKFold strictly grouped by student_id ($N=37$).
2. Inner 3-Fold GroupKFold within training folds for tuning.
3. Historical 6-student test split (`s7, s14, s24, s29, s30, s37`) for historical baseline replication.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import GroupKFold

HISTORICAL_TEST_STUDENTS = ["s7", "s14", "s24", "s29", "s30", "s37"]


def get_student_groups(records: List[Dict[str, Any]]) -> np.ndarray:
    """Returns the student ID grouping array corresponding to each record."""
    return np.array([r["student_id"] for r in records])


def get_outer_cv_splits(records: List[Dict[str, Any]], n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates 5-fold GroupKFold outer splits where every student's 4 parent questions
    remain strictly clustered in either the train or test fold.
    Returns: List of (train_indices, test_indices)
    """
    groups = get_student_groups(records)
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(records, groups=groups))
    return splits


def get_inner_cv_splits(
    records: List[Dict[str, Any]],
    train_indices: np.ndarray,
    n_splits: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates 3-fold GroupKFold inner splits from the outer training indices for hyperparameter tuning.
    """
    sub_records = [records[i] for i in train_indices]
    sub_groups = get_student_groups(sub_records)
    gkf = GroupKFold(n_splits=n_splits)

    inner_splits = []
    for inner_tr, inner_val in gkf.split(sub_records, groups=sub_groups):
        # Map relative sub-indices back to outer training indices
        orig_tr = train_indices[inner_tr]
        orig_val = train_indices[inner_val]
        inner_splits.append((orig_tr, orig_val))
    return inner_splits


def get_historical_exploratory_split(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_indices, test_indices) matching the historical 6-student test partition
    (`s7, s14, s24, s29, s30, s37` as test, remaining 31 students as train).
    """
    test_idx = []
    train_idx = []
    for i, r in enumerate(records):
        if r["student_id"] in HISTORICAL_TEST_STUDENTS:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return np.array(train_idx), np.array(test_idx)
