# Milestone Phase 0: Repository, Data & Provenance Audit

**Audit Date (UTC):** 2026-11-20T16:54:32.655895+00:00  
**Git Commit SHA:** `d29216845898bbc65e374758af875bb1534fd42f`  
**Git Branch:** `main`  
**Working Tree Status:** `M paper/IEEE_TAI_MANUSCRIPT.tex`  
**Auditor:** Antigravity Research Engineer (Acting on behalf of user)

---

## 1. Executive Summary & Verification of Dataset Invariants

| Dataset Unit | Expected | Found On Disk | Match Status | Verification Method |
|---|---:|---:|:---:|---|
| Student Booklets (PDFs) | 37 | 37 | **MATCH** | Exact count of source PDFs in `F:\Automatic answer evaluation dataset\All PDF's` |
| Scanned Pages (JPGs) | 195 | 195 | **MATCH** | Exact count of rendered page JPGs across folders `s1` to `s37` |
| Subquestion Units | 11 | 11 | **MATCH** | 1a, 1b, 1c, 1d (Q1: 8m); 2a, 2b, 2c (Q2: 6m); 3a, 3b, 3c (Q3: 3m); 4 (Q4: 3m) |
| Labelled Subquestion Records | 407 | 407 | **MATCH** | 37 students × 11 subquestions = 407 records in `data/ground_truth/verified_transcriptions.json` |
| Parent-Question Responses | 148 | 148 | **MATCH** | 37 students × 4 parent questions = 148 question responses |
| Total Examination Marks | 20.0 | 20 | **MATCH** | Verified against official IIIT Dharwad DS317 midsem question schema |

All source files have been cryptographically hashed (SHA-256) and cataloged:
- Booklet PDFs manifest: `research_v2/manifests/source_hashes.csv`
- Scanned Pages manifest: `research_v2/manifests/page_inventory.csv`

---

## 2. Granularity of Teacher Marks

Audit of `data/teacher_marks/*.json`:
- Exactly **37 teacher mark files** exist (`s1.json` to `s37.json`).
- **All 37 files contain marks strictly at the parent-question level (Q1, Q2, Q3, Q4).**
- Not a single file in `data/teacher_marks/` contains subquestion-level marks.
- While `data/ground_truth/verified_transcriptions.json` contains a `sub_marks` field, those marks lack independent audit provenance.
- **Rule enforced for Phase 5:** Primary grading evaluation will be strictly conducted at the **parent-question level (Q1–Q4)** to avoid artificially fabricating subquestion gold marks.

---

## 3. Discrepancy & Provenance Audit of Historical Outputs

Every historical reported metric has been audited and classified in `research_v2/00_EVIDENCE_PROVENANCE.csv`:

1. **`research_final/03_ANSWER_REGION_VERIFICATION.csv` (Defect Confirmed):**
   - Contains 407 rows, but **338 rows (83.0%)** contain the full-page placeholder `[0,0,893,1263]`.
   - The same crop hash is duplicated across as many as 11 subquestions.
   - Status column was labeled `automated_verified_consistent` without actual human visual boundary verification.

2. **`research_final/03_TRANSCRIPTION_ADJUDICATION.csv` (Coverage Bottleneck Confirmed):**
   - Only 66 records exist, restricted to the 6 test students (`s7`, `s14`, `s24`, `s29`, `s30`, `s37`).
   - 31 students (341 subquestion responses) have zero adjudicated transcription records in this file.

3. **`scripts/run_clean_research_pipeline.py` (Arbitrary Slicing Confirmed):**
   - Hardcoded to take a central 20%–80% vertical slice of page 2 and compare against `1a + 1b`.
   - This did not constitute genuine answer-crop to subquestion mapping.

4. **`backend/independent_pipeline.py` (Simulated OCR Confirmed):**
   - Simulated OCR metrics by introducing synthetic typos into ground truth transcripts rather than running live PaddleOCR/TrOCR models across all 37 booklets.

5. **`scripts/test_decomposition.py` (Unverified Provenance Confirmed):**
   - Loaded pre-existing `data/student_answers/s37.json` rather than fresh, same-run OCR predictions.
   - Used token overlap `< 0.15` as a heuristic for misrouting, but labeled the output `measured_real`.

6. **Hardware & Calibration Claims:**
   - Peak VRAM of 3.54 GB was an intended specification, not recorded telemetry (execution ran on CPU).
   - Conformal prediction reported 100% coverage with 0 calibration samples.

---

## 4. Preservation Policy

- The entire historical directory `research_final/` remains frozen and untouched as a benchmark artifact.
- All new research artifacts, manifests, live predictions, and reports will be written strictly under `research_v2/` and `experiments/runs/<run_id>/`.
