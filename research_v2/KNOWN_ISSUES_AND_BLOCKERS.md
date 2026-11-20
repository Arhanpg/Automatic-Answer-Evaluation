# Known Issues and Research Blockers (Gate 0 Audit)

## Critical Blockers for Publication-Grade Experiments

1. **Answer Region Ground Truth (Blocker for E2, E3, E4, E5):**
   - 338 of 407 subquestion rows in the repository currently have full-page bounding boxes (`[0,0,893,1263]`) instead of exact handwritten answer boundaries.
   - Running OCR engines on whole uncropped pages and comparing them against subquestion text causes artificial 94% CER errors (as observed in Milestone M2 live PaddleOCR test).
   - **Remedy:** Visual inspection and boundary verification of actual answer sheets is required to extract true answer crops.

2. **Transcription Adjudication Coverage (Blocker for E1, E2):**
   - Only 66 of 407 subquestions (16.2%, covering only the 6 test students) have adjudicated transcripts in `03_TRANSCRIPTION_ADJUDICATION.csv`.
   - The remaining 341 subquestions in `data/ground_truth/verified_transcriptions.json` must be audited directly against the physical scan images.

3. **Granularity of Teacher Marks (Constraint for E6, E7):**
   - Scanned examiner marks in `data/teacher_marks/` exist solely at the parent-question level (Q1, Q2, Q3, Q4).
   - Any subquestion mark in historical CSVs was mathematically partitioned or inferred.
   - **Remedy:** All primary grading benchmarks must report metrics at parent-question level (Q1–Q4) and booklet total (20m total).

4. **Environment Execution Requirements:**
   - PaddleOCR 3.7.0 and PyTorch 2.6.0+cu124 are operational in `F:\Automatic Answer evaluation\.venv`.
   - Tesseract 5.5 is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   - TrOCR is loaded via Hugging Face `microsoft/trocr-small-handwritten`.
