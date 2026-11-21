# Phase 8 Dataset Freeze and Provenance Report

**Generated:** 2026-11-21 15:56:32 UTC  
**Authors:** Arhan Ghosarwade & Abhinav Gumgol  
**Affiliation:** Indian Institute of Information Technology, Dharwad (IIIT Dharwad)  
**Dataset Version:** `Phase 8 Frozen Corpus v3.0`  
**Dataset Freeze Status:** `FROZEN_FOR_EXPERIMENTAL_WORKFLOWS`

---

## 1. Dataset Boundary and Scope
The *Automatic Answer Evaluation* corpus comprises mid-semester examination scripts from an undergraduate computer science course at IIIT Dharwad.

### Canonical Quantities
- **Total Student Booklets ($N$):** 37 (`s1` through `s37`)
- **Total Scanned Page Images ($P$):** 195
- **Total Subquestions per Booklet:** 11 subquestions
  - $Q1$: `1a` (2m), `1b` (2m), `1c` (2m), `1d` (2m) $\rightarrow$ Max: 8.0m
  - $Q2$: `2a` (2m), `2b` (2m), `2c` (2m) $\rightarrow$ Max: 6.0m
  - $Q3$: `3a` (1m), `3b` (1m), `3c` (1m) $\rightarrow$ Max: 3.0m
  - $Q4$: `4` (3m) $\rightarrow$ Max: 3.0m
  - **Exam Total Ceiling:** 20.0m
- **Total Subquestion Records ($N \times 11$):** 407
- **Written Answers:** Exactly 392 (96.3%)
- **Confirmed Omissions:** Exactly 15 (3.7%)
  - Student `s7`: `1b`, `1c`, `2b`, `2c`, `3c` (5 omissions)
  - Student `s18`: `1b`, `1c`, `2b`, `2c`, `3c` (5 omissions)
  - Student `s37`: `1b`, `1c`, `2b`, `2c`, `3c` (5 omissions)

---

## 2. Cryptographic and Provenance Guarantee
1. **Source PDF Integrity:** Every booklet PDF in `AAE_DATA_ROOT/All PDF's/` is cryptographically hashed with SHA-256.
2. **Page Image Integrity:** All 195 page images have validated dimensions and cryptographic SHA-256 digests.
3. **Teacher Mark Verification:** All 148 teacher marks match official faculty grading scorecards with zero subquestion mark synthesis.
4. **Data Isolation:** All raw image scans and unredacted handwritten crops remain strictly local on private storage (`AAE_DATA_ROOT` / `research_v3/crops/`), protected from accidental public git exposure.

---

## 3. Historical Claim Disposition Summary
All 11 claims from the historical Gate 7 manuscript have been formally audited and assigned honest dispositions:
- **C01 (Transcription Accuracy):** `requires_recomputation`
- **C02 (End-to-End Mark Agreement):** `potentially_reusable_after_validation`
- **C03 (Visual Bounding Box Reliability):** `requires_recomputation`
- **C04 (Isolated OCR Character Error Rate):** `genuine_inference_unverified_ground_truth`
- **C05 (Routing Accuracy):** `requires_recomputation`
- **C06 (Component-wise Error Propagation):** `requires_recomputation`
- **C07 (Rubric Dimension Sensitivity):** `potentially_reusable_after_validation`
- **C08 (Confidence Score Calibration):** `historical_reference_only`
- **C09 (Deployment Gate Feasibility):** `invalid_methodology_for_deployment_claim`
- **C10 (Preprocessing Image Filter Utility):** `genuine_inference_unverified_ground_truth`
- **C11 (Corpus Representativeness):** `potentially_reusable_after_validation`

Refer to `05_HISTORICAL_CLAIM_DISPOSITION.csv` for detailed audit findings and required corrective actions.
