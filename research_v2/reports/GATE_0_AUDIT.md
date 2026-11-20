# GATE 0 AUDIT REPORT: Repository, Provenance, and Alignment Audit

**Repository:** `https://github.com/Arhanpg/Automatic-Answer-Evaluation`  
**Working Directory:** `F:\Automatic Answer evaluation\automatic-answer-evaluation\automatic-answer-evaluation`  
**Git Head SHA:** `d29216845898bbc65e374758af875bb1534fd42f`  
**Branch:** `main`  
**Audit Timestamp (UTC):** 2026-11-20T17:00:00Z  
**Gate Status:** **PASSED (All Gate 0 criteria satisfied)**

---

## 1. Physical Dataset Verification & Count Reconciliation

| Unit | Target / Schema | Confirmed On Disk | Status | Verification Detail |
|---|---:|---:|:---:|---|
| **Student Booklets (PDFs)** | 37 | **37** | MATCH | Location: `F:\Automatic answer evaluation dataset\All PDF's\s*.pdf`. SHA-256 cataloged. |
| **Scanned Pages (JPGs)** | 195 | **195** | MATCH | Location: `F:\Automatic answer evaluation dataset\s*\*.jpg`. Dimensions & hashes cataloged. |
| **Parent Questions** | 4 | **4** | MATCH | Q1 (8m), Q2 (6m), Q3 (3m), Q4 (3m) in `data/exam/exam.json` |
| **Parent-Question Responses** | 148 | **148** | MATCH | Exactly 37 booklets × 4 questions = 148 responses |
| **Subquestions** | 11 | **11** | MATCH | 1a, 1b, 1c, 1d, 2a, 2b, 2c, 3a, 3b, 3c, 4 |
| **Subquestion Records** | 407 | **407** | MATCH | Exactly 37 booklets × 11 subquestions = 407 records in `data/ground_truth/verified_transcriptions.json` |
| **Exam Maximum Marks** | 20.0 | **20.0** | MATCH | 8 + 6 + 3 + 3 = 20.0 marks total |

---

## 2. Granularity and Provenance of Teacher Marks

1. **Physical / Stored Marks (`data/teacher_marks/*.json`):**
   - Exactly 37 files exist (`s1.json` to `s37.json`).
   - Every file contains marks strictly at the **parent-question level** (`marks: {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}`).
   - **Zero files** contain teacher-awarded marks at the subquestion level.
2. **Ground Truth File (`verified_transcriptions.json`):**
   - Contains a `sub_marks` field, but its provenance is unadjudicated and inferred from parent totals.
3. **Primary Evaluation Decision:**
   - In accordance with Protocol Section 10.1, primary grading evaluation must be conducted at the **parent-question level (Q1–Q4, total 20.0)** to prevent pseudo-ground truth fabrication.

---

## 3. Forensic Audit of Historical Research Claims & Artifacts

| Artifact | Historical Claim | Provenance Classification | Audit Finding |
|---|---|:---:|---|
| `research_final/03_ANSWER_REGION_VERIFICATION.csv` | 407 verified bounding boxes & crops | **Unverified / Simulated** | **Critical Defect:** 338 of 407 rows (83.0%) contain full-page box `[0,0,893,1263]`. Same crop hash is duplicated across up to 11 subquestions. (e.g., student `s1` subquestion `1a` was assigned page 5, whereas visual inspection proves `1a` is on page 2). |
| `research_final/03_TRANSCRIPTION_ADJUDICATION.csv` | Double-annotated verbatim ground truth | **Human Annotated (Partial)** | **Coverage Defect:** Only 66 records exist, restricted to 6 test students (`s7`, `s14`, `s24`, `s29`, `s30`, `s37`). 31 students (341 records) are missing from adjudication. |
| `backend/independent_pipeline.py` | 27-model leaderboard, OCR CER: Paddle 1.2%, TrOCR 0.38% | **Simulated / Scripted** | Does not run real live OCR inference across all 37 booklets; generates text by injecting synthetic character/word variations into verified transcripts. |
| `experiments/reports/M2_ocr_report.md` | Live PaddleOCR run on s1 & s2: 94.0% CER | **Live Inference (Mismatched Crop)** | Ran live PaddleOCR, but extracted full uncropped pages and compared against subquestion text, resulting in artificial 94% CER. |
| `scripts/run_clean_research_pipeline.py` | Multi-stage preprocessing ablation across test set | **Simulated / Scripted** | Hardcoded central 20%–80% page slice on page 2 only, arbitrarily compared against `1a + 1b`. |
| `scripts/test_decomposition.py` | Stage-wise error decomposition (routing 35% error) | **Unverified** | Loaded pre-existing `data/student_answers/{sid}.json` of unknown vintage; used `< 0.15` token overlap heuristic and labeled it `measured_real`. |
| `README.md` & Telemetry Logs | Peak VRAM 3.54 GB on GPU | **Simulated / Scripted** | Live telemetry records 0.0 MB VRAM; executed on CPU. |
| `outputs/calibration_and_reliability` | Conformal prediction 100% coverage, 90% guarantee | **Simulated / Scripted** | Zero calibration samples recorded in report. |

---

## 4. Preservation & Isolation

- The entire historical directory `research_final/` is frozen and preserved untouched.
- All new research artifacts are being written under `research_v2/` and `experiments/runs/<run_id>/`.
- Manifests generated:
  - `research_v2/manifests/source_hashes.csv`
  - `research_v2/manifests/page_inventory.csv`
  - `research_v2/00_EVIDENCE_PROVENANCE.csv`
  - `research_v2/00_REPOSITORY_AND_DATA_AUDIT.md`
  - `research_v2/KNOWN_ISSUES_AND_BLOCKERS.md`
