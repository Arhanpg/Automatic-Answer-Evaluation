# Model execution audit

The repository contains implementations for classical ML, from-scratch
deep-learning, symbolic, retrieval, LLM, and fine-tuning paths. The tracked
leaderboard does not include an input-modality identifier or a per-sample
prediction/execution manifest. The model matrix consequently marks execution,
weights, and locked-test provenance as incomplete rather than inferring them
from a CSV.

The model leaderboard is retained in `grading_results_by_modality.csv` with an
explicit generated-artifact status. Results from raw OCR, preprocessed OCR,
vision transcription, and verified reference text must be regenerated under a
single fixed protocol before any cross-modality comparison.
