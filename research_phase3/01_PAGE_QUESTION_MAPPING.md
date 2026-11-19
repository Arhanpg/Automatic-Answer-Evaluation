# Phase 3: Page/Question/Answer-Region Mapping

## Result

The source PDFs were read through Tesseract and compared against the repository
references without fabricating transcriptions or answer boundaries.

- 37 source PDFs were located and hashed.
- 195 original pages were rendered and hashed.
- 407 repository-labelled answer/subanswer reference records were exported.
- 195 page images were OCR-extracted with Tesseract.
- 39 subquestions have strong page-reference matches.
- 314 subquestions have weak page-reference matches.
- 54 subquestions remain unresolved.
- Exact answer crop boundaries are still not manually verified.

Generated artifacts:

- `answer_region_manifest.csv`
- `mapping_review_queue.csv`
- `page_review_manifest.csv`
- `answer_region_images/` (195 rendered full pages)
- `full_tesseract_page_ocr.csv`
- `full_tesseract_page_ocr.json`
- `ocr_reference_page_review.csv`

The page images and OCR preview data are generated from the external source
dataset and remain outside the Git-tracked answer crop output. The exact mapping is not publishable until exact crop boundaries and
transcriptions are reviewed. The automated review has now removed the need to
treat every row as equally unknown: strong matches identify likely pages, weak
matches identify candidate pages, and unresolved rows have no reliable evidence.

## Evidence status

The reference transcriptions in `data/ground_truth/verified_transcriptions.json`
are repository-labelled as verified, but no independent annotation protocol,
grader identity, or inter-rater agreement was stored. Consequently, the
transcriptions can be used to audit the repository, but they cannot yet be
presented as independent human ground truth.

## Blocking dependency

Valid CER/WER, exact-expression, and frozen-grader experiments require:

1. A verified question/page region.
2. A verified answer crop.
3. A verified transcription of that crop.
4. The examiner mark and maximum mark.

Until those fields exist, page-level OCR text must not be compared with
question-level text as if they were aligned references.
