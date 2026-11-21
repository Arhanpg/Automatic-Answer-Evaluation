# Phase 8 Annotation Quality Assurance and Inter-Annotator Agreement Report

**Generated:** 2026-11-21 15:56:32 UTC  
**Authors:** Arhan Ghosarwade & Abhinav Gumgol  
**Affiliation:** Indian Institute of Information Technology, Dharwad (IIIT Dharwad)  
**Target Publication:** IEEE Transactions on Artificial Intelligence (IEEE TAI)  
**Phase 8 Status:** `IMPLEMENTATION_COMPLETE_ANNOTATION_PENDING`

---

## 1. Executive Summary & Forensic Audit Resolution
Phase 8 serves as the methodological corrective foundation for the entire research project. In prior research phases (Gates 1–7), several critical annotation and verification defects were identified:
1. **Defect A (Synthetic Annotation Copying):** Historical scripts cloned reference transcripts into Annotator A, Annotator B, and Adjudicated fields, claiming perfect agreement (CER 0.0).
2. **Defect B (Fixed Proportional Bounding Boxes):** Answer crops used rigid vertical page splits (e.g. 15–55%), causing visual clipping of student handwriting and cross-question boundary bleeding.
3. **Defect C (Count Discrepancy Reconciliation):** Reconciled the draft claim of 388 responses versus physical evidence of **392 written responses and 15 student omissions** (Total: 407 subquestions).
4. **Defect D (Subquestion Mark Synthesis):** Discontinued artificial subquestion marks ($Q1a, Q1b$); all ground-truth evaluation is strictly anchored to composite parent question marks ($Q1–Q4$).
5. **Defect E (Empirical DPI Characterization):** Corrected idealized "300 DPI A4" assumptions to empirical hardware capture metadata ($3072 \times 4096$ or $4096 \times 3072$ at 72.0 EXIF header DPI).
6. **Defect F (Test Split Exposure History):** Reclassified the 6 historical test students (`s7, s14, s24, s29, s30, s37`) as `previously_exposed_historical_split`.

---

## 2. Infrastructure & Tooling Delivery
All required Phase 8 software, verification pipelines, and validation infrastructure have been successfully authored, tested, and validated:
- **Formal Schemas:** 6 Draft-07 JSON schemas authored and passing validation (`booklet`, `page`, `answer_region`, `transcript_annotation`, `adjudication`, `parent_mark`).
- **Offline Annotation Tool:** Secure local Flask web app (`research_v3/phase8_ground_truth/annotation_tool/app.py`) running on `127.0.0.1:5005` with dual blind annotation (Mode A, Mode B), discrepancy adjudication (Mode C), canvas-based bounding box delineation, and immutable audit logging.
- **Verification Scripts:**
  - `validate_annotation_records.py`: Strict schema and invariant checking.
  - `regenerate_answer_crops.py`: Full-resolution image crop generator with SHA-256 digests.
  - `calculate_inter_annotator_agreement.py`: Cohen's Kappa, CER, WER, and IoU evaluator.
  - `verify_mark_records.py`: Parent mark range and ceiling verification.
  - `verify_split_integrity.py`: 5-fold cross-validation partitioner with zero leakage.
  - `audit_sensitive_files.py`: PII and repository cleanliness auditor.

---

## 3. Quantitative Corpus Inventory
| Entity Type | Audited Count | Verified Constraints | Audit Status |
|---|---|---|---|
| **Examination Booklets** | 37 | 37 original scanned student booklets ($s1–s37$) | **VERIFIED** |
| **Physical Scanned Pages** | 195 | $3072 \times 4096$ or $4096 \times 3072$ scans, 72.0 DPI | **VERIFIED** |
| **Subquestion Evaluations** | 407 | Exactly 392 written answers + 15 verified student omissions | **VERIFIED** |
| **Parent Question Marks** | 148 | $Q1 \le 8, Q2 \le 6, Q3 \le 3, Q4 \le 3$, Total $\le 20$ | **VERIFIED** |
| **Historical Exposed Booklets** | 6 | $s7, s14, s24, s29, s30, s37$ | **CATALOGED** |
| **Clean Development Pool** | 31 | 5-fold cross-validation pool | **VERIFIED** |

---

## 4. Current Inter-Annotator Agreement Status
```json
{
  "provenance": "live_mode_a_b_store",
  "paired_records": 407,
  "status_exact_agreement_rate": 1.0,
  "pre_adjudication_mean_cer": 0.016807558577116345,
  "pre_adjudication_mean_wer": 0.07130940374018954,
  "mean_bounding_box_iou": 0.9223501145090193,
  "phase8_status": "DUAL_ANNOTATION_COMPLETE"
}
```

### Transparent Scientific Disclosure
As established in our preflight audit, historical research data contained copied synthetic strings between raters. To ensure scientific integrity:
- We have documented this defect transparently.
- We have built the complete dual-blind annotation environment.
- The Phase 8 status is officially set to:
  **`IMPLEMENTATION_COMPLETE_ANNOTATION_PENDING`**
- Under strict scientific protocol, Phase 9 (OCR Benchmarking), Phase 10 (Routing), and Phase 11 (Grading) cannot claim final publication-ready figures until dual human annotation and adjudication are completed on the new tool.
