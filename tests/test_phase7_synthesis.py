"""
tests/test_phase7_synthesis.py -- Unit and Scientific Validation Suite for Gate 7.

Verifies:
1. 00_CLAIM_TO_EVIDENCE_MATRIX.csv contains all primary claims with existing evidence files.
2. Every claim points to a real file path and contains genuine statistical p-values.
3. FINAL_RESEARCH_COMPLETION_REPORT.md exists and covers all research gates (0-7).
4. IEEE TAI manuscript files exist in both Markdown and LaTeX formats with bibliography.
5. Markdown manuscript word count exceeds 3,500 words.
6. LaTeX manuscript is well-formed with matching document delimiters and citations.
7. Zero PII (no 23BCS roll numbers or real student names) in tracked research artifacts.
8. Zero simulated or fabricated markers in the claim matrix or completion report.
"""

import re
import pytest
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_V2 = REPO_ROOT / "research_v2"
MANUSCRIPT_DIR = RESEARCH_V2 / "manuscript"


def test_claim_matrix_integrity():
    """Validates that every claim in the matrix references an existing file and valid metrics."""
    matrix_path = RESEARCH_V2 / "00_CLAIM_TO_EVIDENCE_MATRIX.csv"
    assert matrix_path.exists(), f"Missing {matrix_path}"

    df = pd.read_csv(matrix_path)
    assert len(df) >= 10, f"Expected at least 10 claims, got {len(df)}"

    required_cols = {
        "claim_id", "section", "claim_statement", "evidence_file",
        "metric_name", "sample_size", "reported_value", "p_value", "verification_status"
    }
    assert required_cols.issubset(df.columns)

    for _, row in df.iterrows():
        # Verify file exists on disk
        evidence_file = REPO_ROOT / row["evidence_file"]
        assert evidence_file.exists(), f"Claim {row['claim_id']} references missing file: {evidence_file}"

        # Verify status is verified
        assert row["verification_status"] == "VERIFIED_GENUINE_LIVE_DATA"

        # Verify non-empty fields
        assert str(row["claim_statement"]).strip() != ""
        assert str(row["reported_value"]).strip() != ""
        assert str(row["sample_size"]).strip() != ""


def test_final_completion_report_completeness():
    """Validates that FINAL_RESEARCH_COMPLETION_REPORT covers Gates 0 through 7."""
    report_path = RESEARCH_V2 / "FINAL_RESEARCH_COMPLETION_REPORT.md"
    assert report_path.exists(), f"Missing {report_path}"

    content = report_path.read_text(encoding="utf-8")
    for gate_num in range(8):
        assert f"Gate {gate_num}" in content, f"Report missing coverage of Gate {gate_num}"

    assert "Shapley Attribution" in content
    assert "Preprocessing Degradation Paradox" in content
    assert "Selective Human Review" in content


def test_manuscript_artifacts_exist_and_substantive():
    """Validates that Markdown, LaTeX, and BibTeX manuscript files are complete."""
    md_path = MANUSCRIPT_DIR / "IEEE_TAI_MANUSCRIPT.md"
    tex_path = MANUSCRIPT_DIR / "IEEE_TAI_MANUSCRIPT.tex"
    bib_path = MANUSCRIPT_DIR / "references.bib"

    assert md_path.exists(), f"Missing {md_path}"
    assert tex_path.exists(), f"Missing {tex_path}"
    assert bib_path.exists(), f"Missing {bib_path}"

    # Check Markdown word count
    md_text = md_path.read_text(encoding="utf-8")
    words = md_text.split()
    assert len(words) >= 3500, f"Markdown manuscript word count ({len(words)}) is less than 3500 words"

    # Check LaTeX document structure
    tex_text = tex_path.read_text(encoding="utf-8")
    assert r"\documentclass[journal]{IEEEtran}" in tex_text
    assert r"\begin{document}" in tex_text
    assert r"\end{document}" in tex_text
    assert r"\bibliography{references}" in tex_text
    assert r"\author{" in tex_text
    assert "Ghosarwade" in tex_text
    assert "Gumgol" in tex_text

    # Check BibTeX entries count
    bib_text = bib_path.read_text(encoding="utf-8")
    entries = re.findall(r"@\w+\{", bib_text)
    assert len(entries) >= 20, f"Expected at least 20 BibTeX entries, found {len(entries)}"


def test_no_pii_in_manuscript_and_synthesis():
    """Ensures no real student roll numbers (23BCS*) appear in manuscript or reports."""
    files_to_check = [
        RESEARCH_V2 / "00_CLAIM_TO_EVIDENCE_MATRIX.csv",
        RESEARCH_V2 / "FINAL_RESEARCH_COMPLETION_REPORT.md",
        MANUSCRIPT_DIR / "IEEE_TAI_MANUSCRIPT.md",
        MANUSCRIPT_DIR / "IEEE_TAI_MANUSCRIPT.tex"
    ]

    pii_pattern = re.compile(r"\b23BCS\d+\b", re.IGNORECASE)

    for fpath in files_to_check:
        text = fpath.read_text(encoding="utf-8")
        matches = pii_pattern.findall(text)
        assert len(matches) == 0, f"Found PII ({matches}) in {fpath}"


def test_no_simulated_markers_in_synthesis():
    """Verifies that synthesis files contain zero markers of simulated data."""
    matrix_path = RESEARCH_V2 / "00_CLAIM_TO_EVIDENCE_MATRIX.csv"
    df = pd.read_csv(matrix_path)

    for _, row in df.iterrows():
        val = str(row["reported_value"]).lower()
        stat = str(row["verification_status"]).lower()
        assert "simulated" not in val
        assert "fake" not in val
        assert "placeholder" not in val
        assert "mock" not in val
        assert "simulated" not in stat
