# Phase 3: Recognition-to-Grading Propagation

## Current status

Frozen-grader and modality-specific downstream experiments are blocked by:

- No verified OCR transcriptions aligned to individual answer regions.
- No independently verified crop-to-question mapping.
- No real, complete recognition hypotheses for every locked-test answer.

The repository's historical grading models are not sufficient to prove
recognition-to-grading propagation. They contain generated/simulated values
and synthetic OCR stress conditions.
