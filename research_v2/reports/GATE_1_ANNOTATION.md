# Gate 1 Report: Dataset Manifests & Visual Ground-Truth Verification

**Audit & Completion Date:** 2026-11-21  
**Dataset Name:** IIIT Dharwad DS317 Examination Corpus  
**Repository:** `https://github.com/Arhanpg/Automatic-Answer-Evaluation`  
**Execution Context:** `research_v2/`

---

## 1. Executive Summary & Gate 1 Status
**Gate 1 is PASSED.** All 5 required Phase 1 manifests have been constructed, validated against strict JSON schemas, cryptographically verified, and confirmed with passing unit tests (`tests/test_phase1_manifests.py`).

| Manifest Name | Target Schema | Rows | Verification Checks | Gate 1 Status |
|:---|:---|---:|:---|:---:|
| `booklets.csv` | `schemas/booklet.schema.json` | 37 | All 37 PDFs hashed; splits verified (25 train, 6 val, 6 test); zero leakage | **PASSED** |
| `pages.csv` | `schemas/page.schema.json` | 195 | All 195 pages dimensioned; DPI cataloged; quality tags assigned | **PASSED** |
| `marks.csv` | `schemas/marks.schema.json` | 148 | Strictly at parent-question level (Q1–Q4); max marks sum to 20.0; scores $\le$ max | **PASSED** |
| `transcripts.csv` | `schemas/transcript.schema.json` | 407 | Verbatim transcripts across 11 subquestions; omissions and math flags recorded | **PASSED** |
| `answer_regions.csv`| `schemas/answer_region.schema.json`| 407 | Exact page indices and bounding boxes; crop image hashes recomputed on disk | **PASSED** |

---

## 2. Manifest Inventory & Structure

### A. Booklets (`research_v2/annotations/booklets.csv`)
- **Total Booklets:** 37
- **Splits:**
  - Training Partition: 25 booklets (`s1, s3, s4, s6, s8, s9, s10, s11, s13, s15, s16, s17, s19, s20, s22, s23, s25, s26, s27, s28, s31, s32, s34, s35, s36`)
  - Validation Partition: 6 booklets (`s2, s5, s12, s18, s21, s33`)
  - Locked Test Split: 6 booklets (`s7, s14, s24, s29, s30, s37`)
- **Inclusion Status:** 100% included (37/37).
- **Consent / Ethics:** Institutional educational research exempt; all identifying student names and registration numbers permanently sanitized.

### B. Pages (`research_v2/annotations/pages.csv`)
- **Total Pages:** 195 scanned page images across all 37 booklets.
- **Resolution:** 3072 $\times$ 4096 / 4096 $\times$ 3072 at 72 DPI rendering.
- **Scan Quality Tags:** Documented per page:
  - Cover pages: `cover_sheet;institutional_header;score_table_grid`
  - Internal pages: `student_handwriting;examiner_red_annotations;prose_and_conceptual;matrix_attention_math`

### C. Marks Ground Truth (`research_v2/annotations/marks.csv`)
- **Scoring Level:** Parent-question composite scores ($N = 148$: 37 students $\times$ 4 questions).
- **Maximum Marks:**
  - Q1: 8.0 marks
  - Q2: 6.0 marks
  - Q3: 3.0 marks
  - Q4: 3.0 marks
  - **Booklet Total:** 20.0 marks
- **Integrity Rule Enforced:** In accordance with the Gate 0 audit, subquestion scores are not mathematically partitioned or fabricated. Primary grading evaluation in Phase 5 will be strictly conducted at the parent-question level (Q1–Q4) against physical score box gold marks.

### D. Transcripts (`research_v2/annotations/transcripts.csv`)
- **Subquestion Records:** 407 records ($37 \times 11$).
- **Answer Status Breakdown:**
  - Written: 388 responses (95.3%)
  - Omitted by student: 19 responses (4.7%)
- **Attributes:**
  - Math expressions: 148 responses flagged (`contains_math = True` for 2a, 2b, 2c, 4, etc.)
  - Architecture diagrams: 74 responses flagged (`contains_diagram = True` for 1c, 2b)

### E. Answer Regions & Crop Artifacts (`research_v2/annotations/answer_regions.csv`)
- **Extracted Crops:** Generated and saved into `research_v2/crops/`.
- **Validation:** Bounding boxes `[x0, y0, x1, y1]` and normalized coordinates `[norm_x0, norm_y0, norm_x1, norm_y1]` validated to reside strictly within page boundaries.
- **SHA-256 Verification:** Unit tests recomputed SHA-256 for all extracted crops and confirmed exact match against `crop_sha256` column.

---

## 3. Schema & Integrity Validation Results
All manifests passed strict validation:
- `booklet.schema.json`: PASS (37/37)
- `page.schema.json`: PASS (195/195)
- `marks.schema.json`: PASS (148/148)
- `transcript.schema.json`: PASS (407/407)
- `answer_region.schema.json`: PASS (407/407)
- Unit tests (`tests/test_phase1_manifests.py`): 5/5 PASSED in 4.47s.

---

## 4. Gate 1 Acceptance Criteria
- [x] Every included response has an adjudicated mapping to an exact region and transcript status.
- [x] Every page and region crop hash is reproducible and verified.
- [x] Marks are at a valid parent-question scoring level ($0 \le m \le \text{max}$, sum $\le 20.0$).
- [x] All manifests pass schema validation and test suites.
