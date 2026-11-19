# PDF/OCR/Reference Verification Report

## Executed verification

The source exam PDFs were processed directly:

- 37 student PDFs read.
- 195 student pages rendered and hashed.
- Tesseract OCR executed on all 195 rendered pages.
- 407 repository subquestion references compared with OCR output from every
  candidate page belonging to that student.
- Full OCR text hashes and candidate similarity scores were retained.
- A detailed 407-row actual-OCR-versus-reference report was generated.

## Results

The automated page/reference comparison classified:

- 39 strong page matches.
- 314 weak page matches.
- 54 unresolved matches.

These categories mean only that OCR text was compared with repository reference
text. They do not prove the OCR correctly transcribed the student's answer.

## What was verified

- The PDF is readable and OCR can produce text.
- Candidate answer pages can be ranked for each subquestion.
- OCR output is retained separately from repository reference text.
- OCR/reference similarity is recorded per subquestion.

## What was not verified

- Exact answer-crop boundaries.
- Independent, human-checked transcription of handwritten answers.
- Whether OCR changed or corrected a student's mathematical mistake.
- CER/WER against a verified answer-region reference.
- Whether a cross-check result changes the student's mark.

## Evidence status

The detailed review is an automated cross-check, not a human annotation
workflow. It must not be described as manual human verification or independent
ground truth.
