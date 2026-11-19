# Double-Blind Human Annotation and Adjudication Protocol

## 1. Overview and Objectives
This protocol establishes rigorous, double-blind adjudication standards for transcribing handwritten technical examination booklets and verifying rubric-based mark allocations. The target corpus comprises 37 handwritten exam booklets (195 scanned pages) from an institutional examination on Large Language Models at IIIT Dharwad.

## 2. Examiner Red-Ink Annotation Separation Rule
- **Strict Orthogonal Separation**: Examiner red marks (ticks, crosses, underlined passages, circled numerals, margin mark annotations) must be completely isolated from student-generated answer text.
- **Contamination Hazard**: Teacher annotations inadvertently transcribed by OCR as student responses constitute 'examiner contamination'. This creates artificial score inflation or hallucinated justification tokens. All ground-truth reference transcripts strictly exclude teacher ink.
- **Representation**: Where teacher marks appear in the raw scan, they are recorded in a dedicated metadata column `examiner_red_annotations` with bracketed notation (e.g., `[Examiner: Tick +2.0]`, `[Examiner: Underline 'data sparsity']`).

## 3. Ambiguity and Cursive Adjudication Standards
- **Verbatim Faithful Transcription**: Transcribe exactly what the student wrote, including misspellings (e.g., 'traditiomal', 'datasparsity', 'architerture'), improper punctuation, and informal shorthand.
- **No Over-Correction**: Annotators must never correct a student's technical mistake to match the model answer key. For example, if a student writes `Top-2 selects Word C and Word D` (as in `s29_Q3`), annotators must preserve the incorrect tokens.
- **Crossed-Out Text**: Strikethrough or hatched text is ignored unless explicitly marked 'PTO' or reinstated by the student.
- **Mathematical Notation**: Algebraic expressions, vector matrices, and probability products must preserve exact indices, superscripts, subscripts, and fractional forms (e.g., `P(like|I) = (2+1)/(3+5) = 3/8`).

## 4. Multi-Page and Non-Linear Answer Dispersal
- In written technical assessments, students frequently answer subquestions across non-contiguous pages (e.g., student `s14` wrote Q1a on page 2, and Q1c/Q1d on pages 4–5).
- The human reference transcript aggregates the student's complete answer to each subquestion across all booklet pages while recording the exact source page coordinates.

## 5. Dual-Annotator Adjudication Workflow
- Two independent annotators (Annotator A and Annotator B) transcribe each booklet independently.
- Discrepancies exceeding 0.05 Character Error Rate (CER) or any mark variance are escalated to a Senior Adjudicator.
- Adjudication decisions are logged in `03_TRANSCRIPTION_ADJUDICATION.csv` with explicit rationale.
