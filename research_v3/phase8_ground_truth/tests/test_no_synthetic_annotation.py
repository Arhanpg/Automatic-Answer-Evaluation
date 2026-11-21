"""
research_v3/phase8_ground_truth/tests/test_no_synthetic_annotation.py

Integrity test to verify detection of synthetic / cloned rater annotations:
1. Detects whether historical Annotator A and Annotator B transcript strings
   are identical duplicates across all written answers.
2. Formally asserts that the repository correctly reports this state as
   IMPLEMENTATION_COMPLETE_ANNOTATION_PENDING until real dual-rater records are loaded.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ANN_DIR = REPO_ROOT / "research_v2" / "annotations"
STORE_DIR = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "annotations_store"


def test_detect_historical_synthetic_copying():
    transcripts_file = ANN_DIR / "transcripts.csv"
    assert transcripts_file.exists(), "transcripts.csv missing"
    df = pd.read_csv(transcripts_file)

    written = df[df["answer_status"] == "written"]
    identical = 0
    for _, row in written.iterrows():
        ta = str(row.get("transcript_annotator_a", ""))
        tb = str(row.get("transcript_annotator_b", ""))
        if ta and tb and ta == tb:
            identical += 1

    # In historical research_v2 data, all written transcripts were duplicated
    if identical == len(written):
        # Verify that Phase 8 status file honestly records this defect
        agr_file = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "03_INTER_ANNOTATOR_AGREEMENT.json"
        assert agr_file.exists(), "03_INTER_ANNOTATOR_AGREEMENT.json must exist"
        data = json.loads(agr_file.read_text(encoding="utf-8"))
        assert data.get("phase8_status") == "IMPLEMENTATION_COMPLETE_ANNOTATION_PENDING"
        assert "synthetic" in data.get("audit_finding", "").lower()
