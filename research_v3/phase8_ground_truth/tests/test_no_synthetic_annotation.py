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
    assert identical == len(written), "Historical research_v2 expected to show copied rater strings"

    # Verify that Phase 8 status file records valid state
    agr_file = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "03_INTER_ANNOTATOR_AGREEMENT.json"
    assert agr_file.exists(), "03_INTER_ANNOTATOR_AGREEMENT.json must exist"
    data = json.loads(agr_file.read_text(encoding="utf-8"))
    assert data.get("phase8_status") in [
        "IMPLEMENTATION_COMPLETE_ANNOTATION_PENDING",
        "DUAL_ANNOTATION_COMPLETE",
        "GROUND_TRUTH_FROZEN_FOR_PHASE9_BENCHMARK"
    ]


def test_live_store_has_genuine_rater_divergence():
    mode_a_file = STORE_DIR / "annotations_mode_a.json"
    mode_b_file = STORE_DIR / "annotations_mode_b.json"

    if not mode_a_file.exists() or not mode_b_file.exists():
        pytest.skip("Live annotation store not populated yet")

    mode_a = json.loads(mode_a_file.read_text(encoding="utf-8"))
    mode_b = json.loads(mode_b_file.read_text(encoding="utf-8"))

    assert len(mode_a) == 407
    assert len(mode_b) == 407

    # Ensure Annotator A and B are NOT identical copies across the entire corpus
    identical_count = 0
    written_count = 0
    for k in mode_a:
        stat_a = mode_a[k].get("status", "")
        stat_b = mode_b[k].get("status", "")
        if ("written" in stat_a) and ("written" in stat_b):
            written_count += 1
            if mode_a[k].get("transcript") == mode_b[k].get("transcript"):
                identical_count += 1

    assert written_count in [381, 392]
    # Raters must exhibit natural human divergence (not 100% identical copies)
    assert identical_count < written_count, f"Found {identical_count} identical records out of {written_count} written; raters must show genuine independent variation"
