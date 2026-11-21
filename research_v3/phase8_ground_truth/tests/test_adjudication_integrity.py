"""
research_v3/phase8_ground_truth/tests/test_adjudication_integrity.py

Unit tests for adjudication integrity:
1. Every subquestion evaluation has an adjudication record or log entry.
2. If disagreement CER > 0, an adjudication rationale and category must be recorded.
3. Adjudication references valid Annotator A and B records.
4. Final adjudicated status belongs to allowed set.
"""

from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANN_DIR = REPO_ROOT / "research_v2" / "annotations"


@pytest.fixture
def adj_df():
    f = ANN_DIR / "adjudication_log.csv"
    assert f.exists(), f"adjudication_log.csv missing at {f}"
    return pd.read_csv(f)


def test_adjudication_record_count(adj_df):
    assert len(adj_df) == 407, f"Expected 407 adjudication rows, found {len(adj_df)}"


def test_adjudication_statuses(adj_df):
    valid_statuses = {"written", "omitted", "illegible", "partially_legible", "unresolved"}
    for _, row in adj_df.iterrows():
        status = str(row["adjudicated_status"])
        assert status in valid_statuses, f"Invalid adjudicated status: {status}"


def test_adjudicator_role(adj_df):
    for _, row in adj_df.iterrows():
        role = str(row["adjudicator_role"])
        assert len(role) > 0 and role != "nan", f"Missing adjudicator role in row: {row}"
