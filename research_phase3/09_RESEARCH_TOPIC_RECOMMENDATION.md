# Phase 3: Research Topic Recommendation

## Strongest possible primary contribution

The strongest defensible primary direction is **assessment-critical OCR
propagation**, provided answer-region ground truth can be completed.

It combines:

- Faithful transcription
- Symbol-aware error analysis
- Downstream mark sensitivity
- Selective human review

This is preferable to adaptive preprocessing alone because the present
ablation cannot establish recognition improvement and because preprocessing
ranking is only meaningful with aligned references.

## Supporting directions

1. Transcription faithfulness
2. Grading-sensitive selective prediction

## Blocking evidence

Before IEEE TAI readiness:

- 407 subquestion references require exact page/crop verification.
- Real answer-crop OCR for every training, validation, and locked-test example
  is required.
- Independent transcription adjudication is required.
- Development-only metric and policy fitting must be separated from locked-test
  evaluation.
