# Phase 8 Human Annotation and Adjudication Protocol

**Version:** 3.0  
**Status:** ACTIVE SPECIFICATION  
**Target Study:** Assessment-Critical Handwriting Recognition (IEEE TAI)  
**Authors:** Arhan Ghosarwade, Abhinav Gumgol (*Department of Computer Science and Engineering, IIIT Dharwad*)  
**Date:** November 21, 2026  

---

## 1. Principles of Independent Human Annotation

This protocol governs the reconstruction of ground truth for handwritten university examination booklets. The following core principles are non-negotiable:

1. **Strict Double-Blind Independence:**
   - Every examination response must be independently transcribed by two distinct human annotators: **Annotator A** and **Annotator B**.
   - Annotator A must never view Annotator B's inputs, drafts, or submitted transcriptions.
   - Annotator B must never view Annotator A's inputs, drafts, or submitted transcriptions.
   - Neither annotator may see any automatically generated OCR hypothesis, model answer, or historical reference candidate during independent transcription.
2. **True Visual Boundary Verification:**
   - Every written response must be visually located on the original scanned page image ($3072 \times 4096$ px).
   - Annotators must draw or adjust bounding boxes `[x_min, y_min, x_max, y_max]` to encapsulate the exact handwritten response region, excluding headers, printed questions, and adjacent student answers.
   - Proportional, fixed, or arbitrary central-slice bounding boxes are strictly prohibited.
3. **Faithful Verbatim Transcription (No Error Normalization):**
   - Transcribers must preserve the student's exact written text verbatim.
   - Do NOT correct grammatical errors, spelling mistakes, algebraic sign errors, arithmetic miscalculations, or invalid formulas.
   - Preserve non-standard technical symbols, subscript/superscript notation, and crossed-out or overwritten text.
4. **Transparent Adjudication:**
   - Adjudication occurs only after both Annotator A and Annotator B have submitted completed reviews.
   - The adjudicator evaluates discrepancies, verifies image evidence, selects or synthesizes the canonical transcription, and documents the explicit reason for the decision.

---

## 2. Reviewer Modes & Software Enforcement

The annotation software strictly enforces three operational modes:

### Mode A: Annotator A (Blind)
- Displays the high-resolution scanned page image.
- Displays an empty transcription form, subquestion prompt ID, and bounding box tool.
- Hides all candidate transcripts, Annotator B submissions, and teacher marks.
- Saves record with metadata tag `"annotator_mode": "MODE_A_BLIND"`.

### Mode B: Annotator B (Blind)
- Displays identical raw source images and clean transcription forms.
- Completely isolated from Annotator A's workspace.
- Saves record with metadata tag `"annotator_mode": "MODE_B_BLIND"`.

### Mode C: Adjudication Mode
- Activated only when both Mode A and Mode B submissions exist for a subquestion key.
- Displays side-by-side:
  - Original high-resolution page scan.
  - Annotator A's transcription and bounding box.
  - Annotator B's transcription and bounding box.
  - Historical reference candidate (clearly marked as `unverified_historical_candidate`).
  - Pre-adjudication agreement metrics (strict Character Error Rate and Word Error Rate).
- Requires the adjudicator to select or edit the final canonical text, verify bounding boxes, and log an explicit rationale.

---

## 3. Controlled Answer Statuses

Every response record must be assigned one of the following mutually exclusive statuses:

| Status Code | Description | Criteria |
| :--- | :--- | :--- |
| `written` | Standard written response | Student has written text, equations, or diagrams attempting the subquestion. |
| `omitted_confirmed` | Confirmed student omission | The student wrote nothing for this subquestion anywhere in the booklet. Confirmed by inspecting all pages and continuation areas. |
| `illegible` | Severe physical illegibility | Strokes exist but cannot be deciphered by human readers even under magnification. Must NOT be transcribed as empty string. |
| `partially_legible` | Mixed legibility | Core words are legible, but specific tokens are obscured. Uncertain characters must be bracketed: `[?]` or `[word?]`. |
| `unresolved` | Ambiguous or contested | Boundary or identity cannot be resolved without instructor intervention. Excluded from OCR accuracy benchmarks. |
| `not_applicable` | Inapplicable slot | Structural placeholder for multi-part options not attempted. |

---

## 4. Region Annotation & Coordinate Conventions

### Coordinate Frame
- Pixel coordinates: `[x_min, y_min, x_max, y_max]`
- Origin $(0, 0)$ is at the **top-left** of the image.
- $x$ increases horizontally to the right: $0 \le x_{\min} < x_{\max} \le W$.
- $y$ increases vertically downward: $0 \le y_{\min} < y_{\max} \le H$.
- Normalized coordinates: $\text{norm\_x} = x / W$, $\text{norm\_y} = y / H$, bounded in $[0.0, 1.0]$.

### Multi-Region & Continuation Rules
- If an answer continues across a page boundary or across discontinuous regions on the same page, annotators must create multiple bounding boxes linked by a shared `continuation_group_id` (e.g. `s12_2b_c1`).
- Each continuation segment is cataloged with `region_type`:
  - `student_answer`: Main initial answer block.
  - `continuation_segment`: Subsequent overflow writing on later pages.
  - `diagram_only`: An isolated diagram associated with prose.

---

## 5. Parent-Question Mark Verification Protocol

Official teacher marks are recorded strictly at the parent-question composite level:
- $Q1$: Conceptual NLP & Attention Theory (8.0 marks max)
- $Q2$: Dot-Product Attention & Softmax Derivation (6.0 marks max)
- $Q3$: Discrete Decoding & Sampling Strategies (3.0 marks max)
- $Q4$: Laplace Add-1 Bigram Probability Estimation (3.0 marks max)
- **Total Examination Ceiling:** 20.0 marks max

**Rules:**
1. Ground truth marks are extracted directly from the physical front-cover score table.
2. Two independent reviewers transcribe the marks from physical scan images.
3. Discrepancies between Reviewer A and Reviewer B are adjudicated against the physical score box.
4. **Prohibition of Subquestion Mark Synthesis:** Under no circumstances may a parent score (e.g., 7.0/8.0 on Q1) be mathematically divided across subquestions (1a–1d). All automated grading benchmarks must evaluate at the parent question level.
