# Phase 3 Final Research Audit

## Demonstrated

- 37 source PDFs and 195 pages were hashed and rendered.
- 407 subquestion records and 407 unresolved manual mapping rows were created.
- Tesseract extracted text from all 195 pages.
- Automated page/reference matching classified 407 subquestions:
  39 strong matches, 314 weak matches, and 54 unresolved.
- Real PaddleOCR, Tesseract, and TrOCR execution was retained.
- TrOCR's fixed 0.94 confidence was replaced with generated-token probability.
- A real 28-row Tesseract preprocessing ablation was executed.
- Aggregate OCR metrics were corrected and unit tested.
- All current evidence was labelled as live, generated, or blocked.

## Not demonstrated

- Valid page-to-answer-region ground truth
- Valid OCR CER/WER against human transcription
- Assessment-sensitive error metric
- Adaptive preprocessing ranking on accuracy
- Frozen/modality-specific downstream grader propagation
- Selective grading coverage or workload reduction
- Transcription faithfulness
- Cross-examination generalization
- External literature-derived novelty
- IEEE TAI manuscript readiness
