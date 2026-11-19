# Dataset and ground-truth audit

## Verified observations

- The repository manifest contains 37 booklet IDs and 148 question responses:
  25 train, 6 validation, and 6 locked-test booklets.
- The stored split hash is `41327d9f77e8a40dcc6c027a9250eed79175cc4b6fa2cbdd6f4cc479f40c06e1`. It is reproducible from the manifest
  serialization used by `backend/split_validation.py`; the three sets are disjoint.
- The external source directory was present during this audit and the existing
  corpus audit records 37 PDFs and 195 pages. This audit stores metadata only and
  does not copy scans or expose student names/registration numbers.
- `data/ground_truth/verified_transcriptions.json`, student answers, and teacher
  marks contain 37 IDs. The repository does not contain an independent annotation
  protocol, second-grader labels, or inter-rater agreement.
- `data/models/training_set.jsonl` is empty at audit time.

## Limitations and leakage controls

The unit of independence is the student booklet, not the 148 question rows.
The six locked-test students therefore provide only six independent test groups;
question-level metrics may be useful diagnostics but cannot be interpreted as
148 independent student observations. The stored GroupKFold configuration is
student-grouped, but every reported model result still requires a run log linking
inputs, code revision, seed, and predictions.

The source code in `backend/classical_ml.py` contains a deterministic synthetic
fallback for IDs absent from the gold records. That behavior must be disclosed
and disabled for a real-data claim. Existing reports also describe several
benchmark values as generated or simulated; they remain separate from this audit.

## Required follow-up before a paper

Document the transcription/marking workflow, obtain independent grading or
adjudication, preserve source-to-answer mappings, run deduplication and
near-duplicate checks at student level, and collect external students or a new
exam for validation. Do not tune on the locked test set.
