# Phase 8 Preflight Forensic Audit: Ground-Truth Reconstruction, Annotation Integrity & Data Freeze

**Repository:** `https://github.com/Arhanpg/Automatic-Answer-Evaluation`  
**Authors:** Arhan Ghosarwade, Abhinav Gumgol (*Department of Computer Science and Engineering, IIIT Dharwad*)  
**Audit Baseline Commit:** `678ecf9a8569f16fc357af67f37d880af77c5b13` on branch `main`  
**Working Branch:** `research/phase8-ground-truth-reconstruction`  
**Date:** November 21, 2026  
**Auditor Role:** Senior AI Research Scientist & Software Architect  

---

## 1. Executive Summary

Phase 8 serves as a formal corrective research phase. While prior gates established useful pipeline infrastructure and executed genuine OCR inference, a forensic audit of the data generation code, manifests, and reports reveals methodological defects that must be resolved before further experimental claims are made.

The objective of Phase 8 is to reconstruct a fully verified, scientifically defensible ground-truth dataset. This document records the preflight findings across repository integrity, raw source files, annotation provenance, geometry, split exposure, and privacy risks.

---

## 2. Repository Preflight Status

- **Initial Commit SHA:** `678ecf9a8569f16fc357af67f37d880af77c5b13`
- **Initial Branch:** `main` (Clean working tree)
- **Phase 8 Working Branch:** `research/phase8-ground-truth-reconstruction`
- **Python Environment:** Python 3.12.10 (PyTorch 2.6.0+cu124, CUDA RTX 3050 Laptop GPU, Scipy 1.15.2, Pandas 2.2.3, RapidFuzz 3.12.2, SymPy 1.13.3)
- **Test Suite Status at Baseline:** 36/36 unit tests passing across Phase 1–7 test files.
- **Source Data Access:**
  - Original 37 PDF booklets located at: `F:\Automatic answer evaluation dataset\All PDF's\s1.pdf` .. `s37.pdf`
  - Original 195 high-resolution scanned page images located at: `F:\Automatic answer evaluation dataset\s1` .. `s37`
  - Official teacher marks located at: `data/teacher_marks/s1.json` .. `s37.json`

---

## 3. Forensic Defect Findings

### Defect A: Automated Annotation Copying
- **Location:** `scripts/build_phase1_ground_truth.py`, lines 218–220, 246–248
- **Finding:** The script generated `research_v2/annotations/transcripts.csv` and `adjudication_log.csv` by assigning:
  ```python
  annotator_a = text
  annotator_b = text
  adjudicated = text
  "agreement_cer": 0.0,
  "agreement_wer": 0.0
  ```
- **Methodological Impact:** No actual independent dual-annotator transcription was executed in Phase 1. Inter-annotator agreement metrics of 0.0 CER/WER were artifacts of identical string copying.
- **Corrective Action for Phase 8:** Build a true blind independent annotation protocol (Mode A, Mode B, Mode C) where Annotator A and Annotator B work without seeing each other's transcripts or candidate reference strings.

### Defect B: Fixed Proportional Layout & Hardcoded Crop Coordinates
- **Location:** `scripts/build_phase1_ground_truth.py`, lines 280–292
- **Finding:** Answer crop boundaries were computed using a hardcoded dictionary (`page_sq_layout`) mapping subquestions to fixed pages and fixed percentage vertical intervals:
  - 1a: Page 2, 15% to 55% page height
  - 1b: Page 2, 55% to 95% page height
  - 1c: Page 3, 10% to 52% page height, etc.
- **Methodological Impact:** Handwritten exam booklets have highly variable answer layouts. Students frequently start answers on different pages, write across multiple pages, or skip questions. Proportional cropping results in truncated handwriting, inclusion of extraneous questions, or cropping blank margins.
- **Corrective Action for Phase 8:** Implement a visual boundary annotation tool where human reviewers inspect original high-resolution page scans and manually verify bounding boxes `[x_min, y_min, x_max, y_max]` for all answer regions.

### Defect C: Response Count Discrepancy (388 vs 392)
- **Location:** `research_v2/reports/GATE_1_ANNOTATION.md` vs `research_v2/annotations/transcripts.csv`
- **Finding:**
  - The Gate 1 report stated: 388 written answers, 19 omissions ($388 + 19 = 407$).
  - However, `transcripts.csv` on disk contains: 392 written answers, 15 omissions ($392 + 15 = 407$).
  - The 15 verified omissions are confined to 3 students:
    - `s7`: 1b, 1c, 2b, 2c, 3c (5 omitted)
    - `s18`: 1b, 1c, 2b, 2c, 3c (5 omitted)
    - `s37`: 1b, 1c, 2b, 2c, 3c (5 omitted)
  - `research_v2/crops/` contains exactly 392 crop files (`.png`).
- **Resolution:** The true physical data on disk comprises 392 written answer crops and 15 student omissions. The figure "388" in the Gate 1 report was an uncorrected typographical artifact from early drafting.

### Defect D: Crop Integrity and Bounding Geometry
- **Finding:** All 392 crops in `research_v2/crops/` have matching SHA-256 hashes in `answer_regions.csv`, but their coordinates stem from the proportional formula rather than verified visual bounding boxes. Several crops contain portions of adjacent subquestions or examiner red markings.
- **Corrective Action for Phase 8:** Every crop must be regenerated from verified human bounding box coordinates.

### Defect E: Image Dimensions and DPI Discrepancy
- **Location:** `research_v2/manifests/page_inventory.csv` vs `research_v2/manuscript/IEEE_TAI_MANUSCRIPT.md`
- **Finding:**
  - The physical page image files in `F:\Automatic answer evaluation dataset` have dimensions of **$3072 \times 4096$** (portrait) or **$4096 \times 3072$** (landscape), with EXIF JFIF header DPI recorded as **$72.0 \times 72.0$ DPI**.
  - The manuscript previously stated: *"scanned at 300 DPI, 24-bit RGB, yielding uncompressed page dimensions of approximately $2480 \times 3508$ pixels"*.
  - This 300 DPI claim was an idealized standard A4 formula, not the camera capture metadata.
- **Corrective Action for Phase 8:** Document the exact physical dimensions ($3072 \times 4096$) and EXIF DPI (72 DPI) across all 195 pages. Update all manuscript specifications to reflect empirical facts.

### Defect F: Previous Test-Set Exposure
- **Location:** `research_v2/00_EVIDENCE_PROVENANCE.csv`, Artifact ART-01
- **Finding:** The designated 6 test booklets (`s7, s14, s24, s29, s30, s37`) were repeatedly accessed during early exploratory development scripts (`scripts/run_clean_research_pipeline.py`, `research_final/03_TRANSCRIPTION_ADJUDICATION.csv`).
- **Methodological Impact:** These 6 students cannot be scientifically described as a "completely untouched, virgin held-out test cohort".
- **Corrective Action for Phase 8:** Reclassify the historical test split as `previously_exposed_historical_split`. Mandate 5-fold student-grouped cross-validation for model development, and declare this limitation transparently in the manuscript.

### Defect G: Routing and Uncertainty Calibration Methodological Issues
- **Location:** `scripts/run_phase6_robustness_calibration.py` & `research_v2/calibration/selective_review_curve.csv`
- **Finding:**
  - In `run_phase6_robustness_calibration.py`, the uncertainty score included `(err / 20.0)` (the true gold grading error!) in its formula (`comp_unc = 0.5 * ocr_uncertainty + 0.3 * math_uncertainty + 0.2 * (err / 20.0)`).
  - Reviewed scripts were replaced directly with exact teacher marks (error = 0).
- **Methodological Impact:** Using ground truth error in an uncertainty metric used for selective routing is circular and invalid for real deployment claims.
- **Corrective Action for Phase 8:** Reclassify historical calibration claims as `invalid_methodology_for_deployment_claim`. Mandate that operational uncertainty metrics rely strictly on input-side features (OCR confidence, S-BERT margin, syntactic ambiguity) without leakage of gold marks.

---

## 4. Privacy & Sensitive Data Audit

- **Git-Tracked Files Audited:** 463 files.
- **Institutional Roll Numbers Found:**
  - `scripts/sanitize_remaining_pii.py`: contains `23BCS045` in a regex comment.
  - `tests/test_components.py`: contains `21CS7` in a test fixture.
- **Hardcoded Personal Windows Paths Found:** 26 historical files in `research_audit/` and `research_phase3/` contain hardcoded paths like `F:\Automatic Answer evaluation...`.
- **Git-Tracked Binary Crops:** 41 small PNG files in historical directories.
- **Remediation Protocol:**
  - All new Phase 8 and Phase 9+ code must read private data exclusively via `AAE_DATA_ROOT` environment variable.
  - No personal Windows paths in tracked files.
  - Add explicit `.gitignore` entries for local private data directories and annotation outputs.
  - Maintain student anonymization (`s1` through `s37`).

---

## 5. Dataset Inventory Reconciled

| Entity | Expected / True Count | Status | Notes |
| :--- | :---: | :---: | :--- |
| Student Examination Booklets | 37 | Verified | `s1` to `s37` |
| Original Scanned Page Images | 195 | Verified | $3072 \times 4096$ px, 72 DPI metadata |
| Original PDF Documents | 37 | Verified | Located in `All PDF's/` |
| Parent Question Submissions | 148 | Verified | $Q1$–$Q4$ per student (8m, 6m, 3m, 3m = 20m) |
| Official Teacher Mark Records | 37 | Verified | `data/teacher_marks/s*.json` |
| Granular Subquestion Records | 407 | Verified | $37 \times 11$ subquestions |
| Physical Written Answers | 392 | Reconciled | Corrected from Gate 1 report typo of 388 |
| Confirmed Student Omissions | 15 | Reconciled | Confined to `s7`, `s18`, `s37` |

---

## 6. Phase 8 Decision Gate Status

- **Preflight Audit:** **COMPLETED**
- **Defects Identified & Cataloged:** **COMPLETED**
- **Action Plan:** Proceed immediately to Step 2 (Schemas), Step 3 (Protocol & Data Dictionary), Step 4 (Local Offline Annotation Tool), Step 5 (Integrity & Validation Scripts), and Step 6 (Unit Tests).
- **Overall Phase 8 Status:** `ANNOTATION_IN_PROGRESS`
