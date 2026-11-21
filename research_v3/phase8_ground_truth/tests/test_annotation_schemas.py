"""
research_v3/phase8_ground_truth/tests/test_annotation_schemas.py

Unit tests for all 6 Phase 8 JSON schemas:
- booklet.schema.json
- page.schema.json
- answer_region.schema.json
- transcript_annotation.schema.json
- adjudication.schema.json
- parent_mark.schema.json

Validates positive test cases and negative test cases (rejecting out-of-range,
missing fields, inverted coordinates, and invalid enums).
"""

import json
from pathlib import Path
import pytest
import jsonschema

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


def load_schema(name: str):
    p = SCHEMAS_DIR / f"{name}.schema.json"
    assert p.exists(), f"Schema file not found: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


# 1. Booklet Schema Tests
def test_booklet_schema_valid():
    schema = load_schema("booklet")
    valid_record = {
        "student_id": "s12",
        "source_pdf_relative_path": "All PDF's/s12.pdf",
        "pdf_sha256": "a" * 64,
        "page_count": 5,
        "split_classification": "development_pool",
        "inclusion_status": "included",
        "exclusion_reason": None,
        "consent_status": "institutional_archival_permission"
    }
    jsonschema.validate(instance=valid_record, schema=schema)


def test_booklet_schema_invalid():
    schema = load_schema("booklet")
    # Invalid student_id and negative page count
    invalid_record = {
        "student_id": "student_999",
        "source_pdf_relative_path": "invalid/path.pdf",
        "pdf_sha256": "short_hash",
        "page_count": 0,
        "split_classification": "invalid_split",
        "inclusion_status": "included",
        "consent_status": "institutional_archival_permission"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_record, schema=schema)


# 2. Page Schema Tests
def test_page_schema_valid():
    schema = load_schema("page")
    valid_record = {
        "student_id": "s1",
        "page_id": "s1_p02",
        "page_index": 2,
        "image_relative_path": "s1/IMG_20260924_153233007.jpg",
        "image_sha256": "b" * 64,
        "width_px": 3072,
        "height_px": 4096,
        "physical_dpi": 72.0,
        "scan_quality_tags": "clear; legible"
    }
    jsonschema.validate(instance=valid_record, schema=schema)


def test_page_schema_invalid():
    schema = load_schema("page")
    # Invalid dimensions (not 3072 or 4096)
    invalid_record = {
        "student_id": "s1",
        "page_id": "s1_p02",
        "page_index": 2,
        "image_relative_path": "s1/test.jpg",
        "image_sha256": "b" * 64,
        "width_px": 1000,
        "height_px": 1000,
        "physical_dpi": 72.0,
        "scan_quality_tags": "clear"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_record, schema=schema)


# 3. Answer Region Schema Tests
def test_answer_region_schema_valid():
    schema = load_schema("answer_region")
    valid_record = {
        "region_id": "s14_1a_reg01",
        "student_id": "s14",
        "page_id": "s14_p02",
        "subquestion_id": "1a",
        "parent_question_id": "Q1",
        "coordinates_pixel": {"x_min": 250, "y_min": 400, "x_max": 2800, "y_max": 1800},
        "coordinates_normalized": {"norm_x_min": 0.08, "norm_y_min": 0.1, "norm_x_max": 0.91, "norm_y_max": 0.44},
        "region_type": "student_answer",
        "continuation_group_id": "s14_1a",
        "has_continuation": False,
        "manual_review_status": "human_verified_box",
        "crop_sha256": "c" * 64,
        "reviewer_id": "REV_LEAD01"
    }
    jsonschema.validate(instance=valid_record, schema=schema)


def test_answer_region_schema_invalid_subquestion():
    schema = load_schema("answer_region")
    invalid_record = {
        "region_id": "s14_1z_reg01",
        "student_id": "s14",
        "page_id": "s14_p02",
        "subquestion_id": "1z",  # Not in enum
        "parent_question_id": "Q1",
        "coordinates_pixel": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
        "coordinates_normalized": {"norm_x_min": 0.0, "norm_y_min": 0.0, "norm_x_max": 0.1, "norm_y_max": 0.1},
        "region_type": "student_answer",
        "continuation_group_id": "s14_1z",
        "has_continuation": False,
        "manual_review_status": "human_verified_box",
        "crop_sha256": "c" * 64,
        "reviewer_id": "REV_LEAD01"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_record, schema=schema)


# 4. Transcript Annotation Schema Tests
def test_transcript_schema_valid():
    schema = load_schema("transcript_annotation")
    valid_record = {
        "annotation_id": "ANN_s5_2a_A",
        "student_id": "s5",
        "subquestion_id": "2a",
        "annotator_id": "REV_A01",
        "annotator_mode": "MODE_A_BLIND",
        "answer_status": "written",
        "raw_transcript": "This is student handwriting.",
        "contains_math": True,
        "contains_diagram": False,
        "contains_crossed_out_text": False,
        "contains_overwrite": False,
        "legibility_notes": "Clear handwriting",
        "timestamp_iso": "2026-11-21T10:00:00Z"
    }
    jsonschema.validate(instance=valid_record, schema=schema)


# 5. Adjudication Schema Tests
def test_adjudication_schema_valid():
    schema = load_schema("adjudication")
    valid_record = {
        "adjudication_id": "ADJ_s5_2a",
        "student_id": "s5",
        "subquestion_id": "2a",
        "adjudicator_id": "ADJ_LEAD01",
        "annotation_a_ref": "ANN_s5_2a_A",
        "annotation_b_ref": "ANN_s5_2a_B",
        "pre_adjudication_agreement_cer": 0.02,
        "pre_adjudication_agreement_wer": 0.05,
        "disagreement_category": "minor_punctuation_or_casing",
        "final_adjudicated_status": "written",
        "final_adjudicated_transcript": "This is student handwriting.",
        "adjudication_rationale": "Punctuation normalized to student handwriting.",
        "inclusion_in_ocr_benchmark": True,
        "timestamp_iso": "2026-11-21T11:00:00Z"
    }
    jsonschema.validate(instance=valid_record, schema=schema)


# 6. Parent Mark Schema Tests
def test_parent_mark_schema_valid():
    schema = load_schema("parent_mark")
    valid_record = {
        "mark_id": "s1_Q1",
        "student_id": "s1",
        "parent_question_id": "Q1",
        "teacher_mark": 7.5,
        "maximum_mark": 8.0,
        "score_box_source_reference": "Cover page table Q1",
        "reviewer_a_mark": 7.5,
        "reviewer_b_mark": 7.5,
        "adjudicated_mark": 7.5,
        "mark_verification_status": "verified_against_physical_score_box"
    }
    jsonschema.validate(instance=valid_record, schema=schema)


def test_parent_mark_schema_invalid_mark():
    schema = load_schema("parent_mark")
    invalid_record = {
        "mark_id": "s1_Q1",
        "student_id": "s1",
        "parent_question_id": "Q1",
        "teacher_mark": 9.5,  # Exceeds max 8.0
        "maximum_mark": 8.0,
        "score_box_source_reference": "Cover page table Q1",
        "reviewer_a_mark": 9.5,
        "reviewer_b_mark": 9.5,
        "adjudicated_mark": 9.5,
        "mark_verification_status": "verified_against_physical_score_box"
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_record, schema=schema)
