# Phase 3: Assessment-Sensitive OCR Metrics

## Feasible result

The standard metric implementation was corrected and covered by unit tests.

- Aggregate CER uses total character edit distance divided by total reference
  characters.
- Aggregate WER uses total word edit distance divided by total reference words.
- Empty prediction/reference pairs are explicit.
- Mismatched prediction/reference lists raise `ValueError`.
- Missing prediction/reference lists raise `ValueError`, not a perfect score.
- Whitespace, Unicode symbols, substitution, insertion, deletion, and operator
  cases are tested.

The implementation is in `backend/evaluation.py`.

## Assessment-sensitive analysis status

No defensible assessment-sensitive metric can yet be developed or validated.
Page-level OCR cannot be compared with subquestion references because exact
answer-region crops and page numbers have not been manually verified.

The locked test split must not be used to learn or tune symbol severity weights.
A symbol-aware metric requires:

1. Verified answer-region crops.
2. Student/transcribed reference pairs.
3. OCR hypotheses for those exact crops.
4. Development-only adjudication of grading impact.
5. Locked-test-only final comparison.
