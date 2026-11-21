"""Phase 14: Automated Manuscript Defense and Claim-to-Evidence Verification Test Suite.

Verifies that every empirical claim, number, and statistical test cited in the
manuscript maps directly and without error to verified evidence files on disk.
"""

import csv
import json
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANUSCRIPT_DIR = REPO_ROOT / "manuscript_v3"
CLAIM_MATRIX_PATH = MANUSCRIPT_DIR / "CLAIM_TO_EVIDENCE_MATRIX.csv"
TEX_PATH = MANUSCRIPT_DIR / "IEEE_TAI_MANUSCRIPT.tex"
MD_PATH = MANUSCRIPT_DIR / "IEEE_TAI_MANUSCRIPT.md"
BIB_PATH = MANUSCRIPT_DIR / "references.bib"


@pytest.fixture
def claim_records():
    assert CLAIM_MATRIX_PATH.exists(), f"Claim matrix missing: {CLAIM_MATRIX_PATH}"
    with open(CLAIM_MATRIX_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_claim_matrix_completeness(claim_records):
    """Verify that all 14 claims (C01 through C14) are fully populated."""
    assert len(claim_records) == 14, f"Expected 14 claims, found {len(claim_records)}"
    expected_ids = [f"C{i:02d}" for i in range(1, 15)]
    found_ids = [r["claim_id"] for r in claim_records]
    assert found_ids == expected_ids, f"Claim IDs mismatch: {found_ids} vs {expected_ids}"

    required_columns = [
        "claim_id",
        "section",
        "claim_statement",
        "evidence_file",
        "line_or_row_reference",
        "metric_name",
        "empirical_value",
        "sample_size_N",
        "statistical_test_or_ci",
        "verification_status",
    ]
    for r in claim_records:
        for col in required_columns:
            assert r[col], f"Claim {r['claim_id']} missing value for column {col}"
        assert r["verification_status"] == "verified_empirical"


def test_evidence_files_exist_on_disk(claim_records):
    """Verify that every evidence file referenced in the matrix exists on disk."""
    for r in claim_records:
        raw_files = [f.strip() for f in r["evidence_file"].split("and")]
        for rel_file in raw_files:
            file_path = REPO_ROOT / rel_file
            assert file_path.exists(), f"Evidence file for {r['claim_id']} does not exist: {file_path}"


def test_claim_c01_ground_truth_adjudication():
    """Verify C01: Phase 8 Dual-Rater Agreement."""
    path = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "03_INTER_ANNOTATOR_AGREEMENT.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["paired_records"] == 407
    assert data["status_exact_agreement_rate"] == 1.0
    assert pytest.approx(data["pre_adjudication_mean_cer"], abs=1e-4) == 0.0168
    assert pytest.approx(data["mean_bounding_box_iou"], abs=1e-3) == 0.922


def test_claim_c02_c03_routing_and_rotation():
    """Verify C02 & C03: Phase 10 Rotation Ablation & Hybrid Recovery."""
    rot_path = REPO_ROOT / "research_v3" / "phase10_routing" / "results" / "rotation_ablation.csv"
    with open(rot_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # baseline on rotated
    rot_baseline = [r for r in rows if r["system_id"] == "sheet_parser_baseline" and "Rotated" in r["scan_orientation"]][0]
    assert float(rot_baseline["mean_iou"]) == 0.0793
    assert float(rot_baseline["coverage_rate"]) == 0.0

    sys_path = REPO_ROOT / "research_v3" / "phase10_routing" / "results" / "routing_metrics_by_system.csv"
    with open(sys_path, "r", encoding="utf-8") as f:
        sys_rows = list(csv.DictReader(f))
    hybrid_row = [r for r in sys_rows if r["system_id"] == "hybrid_layout_continuation"][0]
    assert float(hybrid_row["mean_iou_written"]) == 0.9649
    assert float(hybrid_row["subquestion_slot_accuracy"]) == 1.0
    assert float(hybrid_row["cross_question_contamination_rate"]) == 0.0


def test_claim_c04_c05_preprocessing_paradox():
    """Verify C04 & C05: Phase 9 Preprocessing Tests."""
    path = REPO_ROOT / "research_v3" / "phase9_ocr_benchmarking" / "results" / "paired_statistical_tests.csv"
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tess_cer = [r for r in rows if r["engine"] == "tesseract" and r["metric"] == "cer"][0]
    assert float(tess_cer["delta"]) == -1.5042
    assert float(tess_cer["paired_t_stat"]) == -5.7861

    trocr_cer = [r for r in rows if r["engine"] == "trocr" and r["metric"] == "cer"][0]
    assert float(trocr_cer["delta"]) == 0.2075
    assert float(trocr_cer["paired_t_stat"]) == 2.6144
    assert pytest.approx(float(trocr_cer["paired_t_pval"]), abs=1e-4) == 0.0093


def test_claim_c07_c08_c09_c10_c11_factorial_propagation():
    """Verify C07, C08, C09, C10, C11: Phase 11 Factorial ANOVA & Shapley."""
    stage_path = REPO_ROOT / "research_v3" / "phase11_error_propagation" / "results" / "stagewise_summary.csv"
    with open(stage_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    s_a = [r for r in rows if r["stream_id"] == "stream_a"][0]
    s_b = [r for r in rows if r["stream_id"] == "stream_b_trocr"][0]
    s_d_base = [r for r in rows if r["stream_id"] == "stream_d_baseline"][0]
    s_d_hyb = [r for r in rows if r["stream_id"] == "stream_d_hybrid"][0]

    assert float(s_a["total_mae"]) == 6.2973
    assert float(s_b["total_mae"]) == 12.4054
    assert float(s_b["delta_mae_vs_oracle"]) == 6.1081
    assert s_b["shapley_attribution"] == "104.4%"
    assert float(s_d_base["total_mae"]) == 12.7027
    assert float(s_d_hyb["total_mae"]) == 12.4054

    anova_path = REPO_ROOT / "research_v3" / "phase11_error_propagation" / "results" / "anova_decomposition_summary.csv"
    with open(anova_path, "r", encoding="utf-8") as f:
        anova_rows = list(csv.DictReader(f))
    ocr_row = [r for r in anova_rows if "Recognition" in r["Source"]][0]
    assert pytest.approx(float(ocr_row["F"]), abs=1e-1) == 658.03
    assert float(ocr_row["partial_eta_sq"]) == 0.8205


def test_claim_c13_c14_risk_control_and_selective_review():
    """Verify C13 & C14: Phase 12 Calibration and Selective Curves."""
    calib_path = REPO_ROOT / "research_v3" / "phase12_risk_control" / "results" / "calibration_correlation.csv"
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_rows = list(csv.DictReader(f))
    comp_row = [r for r in calib_rows if "Composite" in r["uncertainty_estimator"]][0]
    assert float(comp_row["pearson_r"]) == 0.2715
    assert float(comp_row["spearman_rho"]) == 0.1970

    sub_path = REPO_ROOT / "research_v3" / "phase12_risk_control" / "results" / "subgroup_risk_profile.csv"
    with open(sub_path, "r", encoding="utf-8") as f:
        sub_rows = list(csv.DictReader(f))
    high_u = [r for r in sub_rows if "High" in r["subgroup_name"]][0]
    low_u = [r for r in sub_rows if "Low" in r["subgroup_name"]][0]
    assert float(high_u["mean_absolute_error"]) == 13.175
    assert float(low_u["mean_absolute_error"]) == 12.1471

    curve_path = REPO_ROOT / "research_v3" / "phase12_risk_control" / "results" / "selective_review_curves.csv"
    with open(curve_path, "r", encoding="utf-8") as f:
        curve_rows = list(csv.DictReader(f))
    our_curves = [r for r in curve_rows if "Proposed" in r["triage_strategy"]]
    b20 = [r for r in our_curves if float(r["review_budget_pct"]) == 20.0][0]
    b50 = [r for r in our_curves if float(r["review_budget_pct"]) == 50.0][0]
    b75 = [r for r in our_curves if float(r["review_budget_pct"]) == 75.0][0]

    assert float(b20["residual_MAE"]) == 10.0676
    assert float(b20["workload_saved_pct"]) == 81.1
    assert float(b50["residual_MAE"]) == 6.3649
    assert float(b75["residual_MAE"]) == 2.3378
    assert float(b75["residual_QWK"]) == 0.6436


def test_tex_and_md_presence_of_key_metrics():
    """Verify that both TeX and MD documents explicitly cite the key empirical metrics."""
    assert TEX_PATH.exists()
    assert MD_PATH.exists()
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    md_text = MD_PATH.read_text(encoding="utf-8")

    key_tokens = [
        "658.03",  # ANOVA F
        "0.8205",  # partial eta^2
        "104.4%",  # Shapley
        "0.9649",  # Hybrid IoU
        "0.0793",  # Rotated baseline IoU
        "1.1687",  # TrOCR Raw CER
        "1.3762",  # TrOCR Prep CER
        "12.4054", # Stream B MAE
        "6.2973",  # Stream A MAE
        "10.0676", # 20% budget residual MAE
        "81.1%",   # Workload saved
        "0.6436",  # 75% budget QWK
    ]

    for tok in key_tokens:
        tex_tok = tok.replace("%", "\\%")
        assert tex_tok in tex_text, f"Token {tex_tok} missing in IEEE_TAI_MANUSCRIPT.tex"
        assert tok in md_text, f"Token {tok} missing in IEEE_TAI_MANUSCRIPT.md"


def test_bib_citation_keys():
    """Verify that all citation keys used in TeX exist in references.bib."""
    assert BIB_PATH.exists()
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))

    tex_text = TEX_PATH.read_text(encoding="utf-8")
    cite_matches = re.findall(r"\\cite\{([^}]+)\}", tex_text)
    used_keys = set()
    for m in cite_matches:
        for k in m.split(","):
            used_keys.add(k.strip())

    missing_keys = used_keys - bib_keys
    assert not missing_keys, f"Citation keys in TeX missing from bib: {missing_keys}"
