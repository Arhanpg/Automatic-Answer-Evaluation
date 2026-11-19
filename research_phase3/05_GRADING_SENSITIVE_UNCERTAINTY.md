# Phase 3: Grading-Sensitive Uncertainty

## Current status

Selective grading is not experimentally valid until real answer-region OCR
hypotheses and locked-test mark labels are available.

The Phase 2 TrOCR field was corrected from a fixed 0.94 confidence placeholder
to generation-sequence token probability only if the model exposes
`output_scores`. PaddleOCR and Tesseract confidence scales remain engine-
specific and cannot be averaged as calibrated probabilities.

No automatic coverage, review reduction, or workload claim is made.
