# Milestone M10: Reliability and Uncertainty Report

## Conformal artifact

The saved conformal JSON contains 24 test records, nominal 90% coverage,
1.0 mark mean interval width, and finite-sample quantile `0.5`.

The artifact reports `n_calibration=0`, even though it reports
100% test-record coverage. Therefore:

- It cannot support a correctly calibrated 90% conformal guarantee.
- A valid deployment study requires a real held-out calibration partition.
- The 1.0 mark intervals demonstrate an output format and routing rule, not
  a proven coverage guarantee.

## OCR-to-grading propagation

The four-stage OCR artifact reports CER 3.8%, WER 6.2%, and grading
propagation MAE approximately 0.44 in the M10 report. Raw PaddleOCR is listed
as MAE 2.35. These values should be checked by a live end-to-end run before
being presented as validated propagation measurements.

## Human review routing

Low confidence and OCR disagreement can populate a review queue. The queue
should be used before training labels are accepted. No fairness claim is valid
without audited concise and verbose answer pairs. The saved audit reports zero
fairness samples and must not be described as a completed fairness experiment.
