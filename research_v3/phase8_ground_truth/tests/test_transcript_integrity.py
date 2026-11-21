"""
research_v3/phase8_ground_truth/tests/test_transcript_integrity.py

Unit tests for transcript records:
1. Exactly 407 transcript records (37 students * 11 subquestions).
2. Exactly 392 written transcripts and 15 omitted transcripts.
3. Written records have non-empty transcripts.
4. Omitted records have empty transcripts and appropriate status flags.
5. All 11 expected subquestions (1a-1d, 2a-2c, 3a-3c, 4) present per student.
"""

from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANN_DIR = REPO_ROOT / "research_v2" / "annotations"
EXPECTED_SUBQS = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b", "3c", "4"]


@pytest.fixture
def transcripts_df():
    f = ANN_DIR / "transcripts.csv"
    assert f.exists(), f"transcripts.csv missing at {f}"
    return pd.read_csv(f)


def test_transcript_record_counts(transcripts_df):
    assert len(transcripts_df) == 407, f"Expected 407 records, found {len(transcripts_df)}"
    assert transcripts_df["student_id"].nunique() == 37, f"Expected 37 students, found {transcripts_df['student_id'].nunique()}"


def test_subquestion_completeness(transcripts_df):
    for sid, group in transcripts_df.groupby("student_id"):
        subqs = sorted(group["subquestion_id"].tolist())
        assert subqs == sorted(EXPECTED_SUBQS), f"Student {sid} missing subquestions: {subqs}"


def test_omission_integrity(transcripts_df):
    omitted = transcripts_df[transcripts_df["answer_status"] == "omitted"]
    assert len(omitted) == 15, f"Expected exactly 15 omissions, found {len(omitted)}"

    # All omissions must have empty transcripts
    for _, row in omitted.iterrows():
        ta = str(row.get("transcript_annotator_a", ""))
        tb = str(row.get("transcript_annotator_b", ""))
        assert ta == "" or ta == "nan", f"Omitted answer has non-empty annotator_a: {ta}"
        assert tb == "" or tb == "nan", f"Omitted answer has non-empty annotator_b: {tb}"


def test_written_integrity(transcripts_df):
    written = transcripts_df[transcripts_df["answer_status"] == "written"]
    assert len(written) == 392, f"Expected exactly 392 written records, found {len(written)}"

    for _, row in written.iterrows():
        t = str(row.get("adjudicated_transcript", ""))
        assert len(t.strip()) > 0 and t != "nan", f"Written answer has empty adjudicated transcript: {row['student_id']}_{row['subquestion_id']}"
