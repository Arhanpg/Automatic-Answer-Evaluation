# Phase 8 Data Dictionary: Ground-Truth Entities & Schemas

**Target Corpus:** 37 Institutional Examination Booklets (195 Pages, 407 Subquestions, 148 Parent Questions)  
**Schema Standard:** JSON Schema Draft-07 compliant  
**Package:** `research_v3/phase8_ground_truth/`  

---

## 1. Entity Architecture

The Phase 8 data model enforces a normalized relational structure:
- **Booklet (`booklet.schema.json`)**: High-level metadata for each of the 37 examination scripts.
- **Page (`page.schema.json`)**: Physical scanned page image specifications and cryptographic hashes.
- **Answer Region (`answer_region.schema.json`)**: Exact pixel-level and normalized bounding box coordinates for each written answer segment.
- **Transcript Annotation (`transcript_annotation.schema.json`)**: Independent verbatim human transcriptions from Mode A (Annotator A) and Mode B (Annotator B).
- **Adjudication (`adjudication.schema.json`)**: Canonical adjudicated transcripts and resolution logs for discrepancies.
- **Parent Mark (`parent_mark.schema.json`)**: Official verified teacher marks at the composite question level ($Q1$–$Q4$).

---

## 2. Entity Field Specifications

### 2.1 Booklet Entity (`booklet.schema.json`)
| Field Name | Type | Constraints / Format | Semantic Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | `^s([1-9]\|[1-2][0-9]\|3[0-7])$` | Unique pseudonymous identifier for the student booklet (`s1`..`s37`). |
| `source_pdf_relative_path` | String | `^All PDF's/s[0-9]+\.pdf$` | Relative path to the original uncompressed PDF document in dataset root. |
| `pdf_sha256` | String | 64-char hex string | Cryptographic SHA-256 digest of the PDF file. |
| `page_count` | Integer | Min: 3, Max: 12 | Total number of physical scanned pages. |
| `split_classification` | String | Enum: `development_pool`, `previously_exposed_historical_split` | Split designation reflecting exploratory exposure history. |
| `inclusion_status` | String | Enum: `included`, `excluded_pending_audit`, `withdrawn` | Final status of booklet in empirical benchmarks. |
| `exclusion_reason` | String / Null | Nullable string | Explicit rationale for exclusion if applicable. |
| `consent_status` | String | Enum: `institutional_archival_permission`, `private_educational_record_restricted` | Data privacy and institutional access tier. |

### 2.2 Page Entity (`page.schema.json`)
| Field Name | Type | Constraints / Format | Semantic Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | Foreign key to Booklet | Associated booklet identifier. |
| `page_id` | String | `^s[0-9]+_p[0-9]{2}$` | Primary key combining student ID and 2-digit 1-indexed page number (`s1_p01`). |
| `page_index` | Integer | Min: 1, Max: 15 | Physical 1-based page sequence within booklet. |
| `image_relative_path` | String | Path string | Relative path to raw image from `AAE_DATA_ROOT` (e.g. `s1/IMG_...jpg`). |
| `image_sha256` | String | 64-char hex string | Cryptographic SHA-256 digest of original raw page image. |
| `width_px` | Integer | Enum: [3072, 4096] | Native pixel width of image sensor capture. |
| `height_px` | Integer | Enum: [3072, 4096] | Native pixel height of image sensor capture. |
| `physical_dpi` | Number | Typically 72.0 | Hardware EXIF JFIF header DPI metadata value. |
| `scan_quality_tags` | String | Semicolon-delimited | Descriptive tags (e.g. `cover_page;clean;red_annotations;ruled_grid`). |

### 2.3 Answer Region Entity (`answer_region.schema.json`)
| Field Name | Type | Constraints / Format | Semantic Description |
| :--- | :--- | :--- | :--- |
| `region_id` | String | `^s[0-9]+_(1[a-d]\|2[a-c]\|3[a-c]\|4)_reg[0-9]+$` | Primary key for region segment (`s1_1a_reg1`). |
| `student_id` | String | Foreign key to Booklet | Associated booklet ID. |
| `page_id` | String | Foreign key to Page | Page ID containing this bounding box. |
| `subquestion_id` | String | Enum: `1a`..`1d`, `2a`..`2c`, `3a`..`3c`, `4` | Subquestion slot. |
| `parent_question_id` | String | Enum: `Q1`, `Q2`, `Q3`, `Q4` | Composite parent question identifier. |
| `coordinates_pixel` | Object | `{x_min, y_min, x_max, y_max}` | Absolute pixel coordinates on page image (top-left origin). |
| `coordinates_normalized` | Object | `{norm_x_min, norm_y_min, norm_x_max, norm_y_max}` | Coordinates scaled to $[0.0, 1.0]$. |
| `region_type` | String | Enum: `student_answer`, `diagram_only`, `continuation_segment`, `omitted_slot` | Functional segment type. |
| `continuation_group_id` | String | E.g. `s1_1a_c1` | Identifier linking multi-segment responses. |
| `has_continuation` | Boolean | True/False | Indicator if answer overflows to other regions. |
| `manual_review_status` | String | Enum: `human_verified_box`, `adjudicated_box`, `omitted_slot_confirmed`, `unresolved_boundary` | Verification assurance level. |
| `crop_sha256` | String | 64-char hex or `OMITTED_BY_STUDENT` | Cryptographic SHA-256 hash of extracted crop image. |
| `reviewer_id` | String | E.g. `REV_01`, `ADJ_LEAD` | Pseudonym of human reviewer who verified coordinates. |

### 2.4 Transcript Annotation Entity (`transcript_annotation.schema.json`)
| Field Name | Type | Constraints / Format | Semantic Description |
| :--- | :--- | :--- | :--- |
| `annotation_id` | String | `^ANN_s[0-9]+_[0-9a-z]+_[AB]$` | Primary key (`ANN_s1_1a_A`). |
| `student_id` | String | Foreign key to Booklet | Associated booklet ID. |
| `subquestion_id` | String | Subquestion slot | Target question slot. |
| `annotator_id` | String | `^REV_[A-Z0-9_]+$` | Pseudonymous rater ID. |
| `annotator_mode` | String | Enum: `MODE_A_BLIND`, `MODE_B_BLIND` | Independent operational mode. |
| `answer_status` | String | Enum: `written`, `omitted_confirmed`, `illegible`, `partially_legible`, `unresolved` | Independent categorization. |
| `raw_transcript` | String | Text | Verbatim student transcription. |
| `contains_math` | Boolean | True/False | Flag for mathematical equations or matrices. |
| `contains_diagram` | Boolean | True/False | Flag for architectural diagrams. |
| `contains_crossed_out_text` | Boolean | True/False | Flag for struck-through lines. |
| `contains_overwrite` | Boolean | True/False | Flag for overwritten characters. |
| `legibility_notes` | String | Text | Rater observations on ambiguities. |
| `timestamp_iso` | String | ISO 8601 UTC timestamp | Submission timestamp. |

### 2.5 Adjudication Entity (`adjudication.schema.json`)
| Field Name | Type | Constraints / Format | Semantic Description |
| :--- | :--- | :--- | :--- |
| `adjudication_id` | String | `^ADJ_s[0-9]+_[0-9a-z]+$` | Primary key (`ADJ_s1_1a`). |
| `student_id` | String | Foreign key | Associated booklet ID. |
| `subquestion_id` | String | Subquestion slot | Question slot. |
| `adjudicator_id` | String | `^ADJ_[A-Z0-9_]+$` | Pseudonymous adjudicator ID. |
| `annotation_a_ref` | String | Foreign key | Reference to Mode A annotation ID. |
| `annotation_b_ref` | String | Foreign key | Reference to Mode B annotation ID. |
| `pre_adjudication_agreement_cer` | Number | $\ge 0.0$ | Strict Character Error Rate between A and B prior to adjudication. |
| `pre_adjudication_agreement_wer` | Number | $\ge 0.0$ | Strict Word Error Rate between A and B prior to adjudication. |
| `disagreement_category` | String | Enum: `exact_agreement`, `minor_punctuation_or_casing`, `lexical_token_discrepancy`, `mathematical_symbol_discrepancy`, `status_discrepancy`, `unresolved` | Disagreement taxonomy. |
| `final_adjudicated_status` | String | Enum: `written`, `omitted_confirmed`, `illegible`, `partially_legible`, `unresolved` | Authoritative consensus status. |
| `final_adjudicated_transcript` | String | Text | Canonical adjudicated text. |
| `adjudication_rationale` | String | Text | Documentation for adjudicator choice. |
| `inclusion_in_ocr_benchmark` | Boolean | True/False | Flag for whether transcript enters OCR evaluation. |
| `timestamp_iso` | String | ISO 8601 UTC timestamp | Adjudication timestamp. |

### 2.6 Parent Mark Entity (`parent_mark.schema.json`)
| Field Name | Type | Constraints / Format | Semantic Description |
| :--- | :--- | :--- | :--- |
| `mark_id` | String | `^s[0-9]+_Q[1-4]$` | Primary key (`s1_Q1`). |
| `student_id` | String | Foreign key | Associated booklet ID. |
| `parent_question_id` | String | Enum: `Q1`, `Q2`, `Q3`, `Q4` | Composite question ID. |
| `teacher_mark` | Number | $[0.0, 8.0]$ | Verified mark awarded by official course examiner. |
| `maximum_mark` | Number | Enum: [3.0, 6.0, 8.0] | Maximum allowable marks ($Q1=8, Q2=6, Q3=3, Q4=3$). |
| `score_box_source_reference` | String | Text | Physical location on booklet front page score grid. |
| `reviewer_a_mark` | Number | $[0.0, 8.0]$ | First independent mark transcription. |
| `reviewer_b_mark` | Number | $[0.0, 8.0]$ | Second independent mark transcription. |
| `adjudicated_mark` | Number | $[0.0, 8.0]$ | Consensus mark after physical score box audit. |
| `mark_verification_status` | String | Enum: `verified_against_physical_score_box`, `adjudicated_discrepancy`, `unresolved` | Audit assurance level. |
