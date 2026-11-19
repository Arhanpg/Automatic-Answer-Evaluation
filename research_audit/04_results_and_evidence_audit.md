# Results and evidence audit

The repository documentation already distinguishes recorded implementation
outputs from simulated/calibrated benchmark artifacts. This audit confirms
that distinction and records the following high-risk inconsistencies:

- The hardware registry contains historical entries with both zero and nonzero
  VRAM; current evidence does not establish a live GPU run.
- The registry contains contradictory historical classical-model metrics.
- The M10 artifact reports apparent test coverage but the report states zero
  calibration records, so it is not a valid calibrated guarantee.
- The independent OCR benchmark names external engines while its report says
  text variations were scripted from verified transcriptions.

No existing result or source dataset was overwritten. The CSVs in this directory
are evidence-index artifacts and are not replacement benchmark results.
