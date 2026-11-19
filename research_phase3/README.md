# Phase 3 Research Artifacts

The completed items are:

- Full 195-page source render and hash manifest
- Full Tesseract OCR text extraction for 195 pages
- Page-reference cross-check for 407 subquestion records
  - 39 strong matches
  - 314 weak matches
  - 54 unresolved matches
- Detailed actual-OCR-versus-reference review:
  `detailed_ocr_reference_review.csv`
- 407 unresolved answer-reference mapping rows
- Real Phase 2 PaddleOCR, Tesseract, and TrOCR execution records
- Real TrOCR token-level probability replacement
- 28-row Tesseract preprocessing ablation
- Corrected and tested aggregate OCR metrics

The audit is not complete until the exact answer regions and references are
verified. This repository intentionally does not report false CER/WER values
for page-level text compared with question-level references.
