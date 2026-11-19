# OCR error taxonomy

This taxonomy is an audit schema, not a count of independently human-labelled
errors. The Phase 2 run stores output hashes and text lengths because the
available repository records do not align each rendered page with a verified
question transcription.

| Category | Definition | Evidence status |
|---|---|---|
| Red-ink removal | Examiner red pixels remain in student OCR input, or student ink is removed | Algorithm implemented; sample visual review not independently adjudicated |
| Gridline/deskew artifact | Table lines remain, or rotation introduces border/stroke artifacts | Algorithm executed; failure labels not human-validated |
| Denoising loss | Minus sign, dot, decimal, fraction bar, superscript, or subscript is removed | Risk identified from parameters; no symbol-level labels |
| Binarization distortion | Sauvola output merges or breaks strokes or mathematical structures | Algorithm executed; no ground-truth mask |
| Insertion | OCR adds characters or words absent from reference | Requires aligned reference; not counted in Phase 2 |
| Deletion | OCR omits characters or words present in reference | Requires aligned reference; not counted in Phase 2 |
| Substitution | OCR changes a character, operator, or word | Requires aligned reference; not counted in Phase 2 |
| Mathematical-operator error | `+`, `-`, `/`, `^`, vector, matrix, or delimiter is wrong | Text metrics alone are insufficient |
| Mathematical-equivalence ambiguity | Text differs while expression may be equivalent, or text matches while meaning differs | Requires subject-expert adjudication |
| Page/question mismatch | OCR output is compared with the wrong answer reference | Current repository mapping is insufficient; block metric claim |

No category should be reported as a measured error rate until page/question
alignment and independent human adjudication are added.
