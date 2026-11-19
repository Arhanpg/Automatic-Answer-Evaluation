# Preprocessing and OCR audit

The implementation in `backend/independent_pipeline.py` contains the reported
four stages: HSV red-ink isolation, grayscale/grid suppression/deskew,
Fast Non-Local Means denoising, and Sauvola binarization. The configured values
include `h=10`, template/search windows 7/21, and Sauvola window 25, `k=0.2`.

The repository contains representative preprocessing images under
`static/images/preprocessing_samples/`, but this audit does not claim a complete
independent visual review of all source pages or human verification of symbols.

The tracked M2 report explicitly says the independent PaddleOCR, Tesseract, and
VLM tables are scripted/calibrated from verified text. The live CSV contains only
two rows and very high raw CER/WER. Those tracks are therefore kept separate in
`ocr_track_comparison.csv` and must not be combined into a preprocessing
improvement claim. Mathematical equivalence is not established by textual CER/WER.
