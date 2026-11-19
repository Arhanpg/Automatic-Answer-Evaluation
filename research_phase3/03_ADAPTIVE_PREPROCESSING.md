# Phase 3: Adaptive Preprocessing

## Executed experiment

A real Tesseract ablation was executed on page 1 of `s1`, `s2`, `s18`, and
`s29` for:

- Original image
- Red-ink removal
- Red-ink removal plus deskew
- Grid suppression
- Denoising
- Sauvola binarization
- Complete existing pipeline

The 28 output rows are recorded in `preprocessing_ablation.csv`.

## Scientific limitation

No CER, WER, or mathematical-symbol improvement is reported because the
repository does not contain verified question/subquestion text aligned to
answer regions. Character and word counts describe OCR output volume only and
must not be interpreted as accuracy.

## Adaptive policy status

Adaptive preprocessing selection is not yet trainable without aligned OCR
references and development/validation splits. The policy must not select the
best condition using test text. At this stage only the ablation experiment
infrastructure is established.
