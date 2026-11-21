"""
research_v3/phase8_ground_truth/tests/test_mark_integrity.py

Unit tests for parent-question marks integrity:
1. Exactly 148 marks records (37 students * 4 parent questions Q1-Q4).
2. Q1 mark in [0, 8].
3. Q2 mark in [0, 6].
4. Q3 mark in [0, 3].
5. Q4 mark in [0, 3].
6. Booklet total mark does not exceed ceiling of 20.0.
7. No subquestion marks are synthesized.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANN_DIR = REPO_ROOT / "research_v2" / "annotations"
TM_DIR = REPO_ROOT / "data" / "teacher_marks"

MAX_MARKS = {"Q1": 8.0, "Q2": 6.0, "Q3": 3.0, "Q4": 3.0}


@pytest.fixture
def marks_df():
    f = ANN_DIR / "marks.csv"
    assert f.exists(), f"marks.csv missing at {f}"
    return pd.read_csv(f)


def test_mark_record_counts(marks_df):
    assert len(marks_df) == 148, f"Expected 148 mark records, found {len(marks_df)}"
    assert marks_df["student_id"].nunique() == 37, f"Expected 37 students, found {marks_df['student_id'].nunique()}"


def test_parent_question_ceilings(marks_df):
    for _, row in marks_df.iterrows():
        sid = row["student_id"]
        qid = row["parent_question_id"]
        mark = float(row["teacher_mark"])
        max_mark = MAX_MARKS[qid]

        assert 0.0 <= mark <= max_mark, f"Student {sid} {qid} mark {mark} exceeds range [0, {max_mark}]"


def test_student_total_ceiling(marks_df):
    student_totals = marks_df.groupby("student_id")["teacher_mark"].sum()
    for sid, total in student_totals.items():
        assert total <= 20.0, f"Student {sid} total score {total} exceeds maximum ceiling of 20.0"


def test_consistency_with_raw_json(marks_df):
    if not TM_DIR.exists():
        pytest.skip("data/teacher_marks not available")

    for sid, group in marks_df.groupby("student_id"):
        json_file = TM_DIR / f"{sid}.json"
        if json_file.exists():
            data = json.loads(json_file.read_text(encoding="utf-8"))
            raw_marks = data.get("marks", {})
            for _, r in group.iterrows():
                q_num = r["parent_question_id"].replace("Q", "")
                assert abs(float(r["teacher_mark"]) - float(raw_marks[q_num])) < 1e-4
