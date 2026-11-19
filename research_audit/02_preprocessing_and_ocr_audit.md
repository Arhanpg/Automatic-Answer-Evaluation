# Preprocessing and OCR audit

## Implementation verification

The implementation in `backend/independent_pipeline.py` contains the reported
four stages: HSV red-ink isolation, grayscale/grid suppression/deskew,
Fast Non-Local Means denoising, and Sauvola binarization. The configured values
include `h=10`, template/search windows 7/21, and Sauvola window 25, `k=0.2`.
The separate `backend/red_ink_extractor.py` path uses a different red saturation
threshold and a different width cap, so these two paths must not be silently
treated as identical implementations.

## Actual Phase 2 run

`run_phase2_audit.py` rendered page 1 from `s1`, `s2`, `s18`, and `s29` using
the local source PDFs. It generated all six stage images for each sample in a
local temporary directory and stored only source/image hashes and measurements
in `phase2_preprocessing_samples.csv`. Observed red-pixel fractions were
0.000171–0.028397 and estimated skew ranged from 0 to 1.4991 degrees.
These are sample measurements, not corpus-wide performance claims.

Tesseract was executed locally on raw rendered pages and Sauvola outputs.
`phase2_actual_ocr_runs.csv` records output hashes and lengths, but does not
report CER/WER because the repository has no page-to-question transcription
mapping. PaddleOCR was not installed, and no VLM model was downloaded or run.
No examination images were copied into the repository.

## Evidence restrictions

The tracked M2 report explicitly says the independent PaddleOCR, Tesseract, and
VLM tables are scripted/calibrated from verified text. The live CSV contains only
two rows and very high raw CER/WER. Those tracks are therefore kept separate in
`ocr_track_comparison.csv` and must not be combined into a preprocessing
improvement claim. Mathematical equivalence is not established by textual
CER/WER.

The metric audit also found that `backend/independent_pipeline.py` normalizes
CER by reference length, while `backend/evaluation.py` uses the maximum of
hypothesis and reference lengths. The insertion case therefore produces
different CER values (1/3 versus 1/4). A future benchmark must select and
document one metric definition before comparing tracks.
