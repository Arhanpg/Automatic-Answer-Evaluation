# Phase 8 Local Human Annotation and Adjudication Tool

## 1. Overview
The **Phase 8 Annotation Tool** is an offline, privacy-preserving web application designed specifically to correct historical annotation and bounding-box defects in the *Automatic Answer Evaluation* study.

It provides a triple-mode workflow:
- **Mode A (Blind Annotator 1)**: Independent visual inspection, answer bounding-box delineation, and verbatim handwriting transcription without seeing Annotator B or reference data.
- **Mode B (Blind Annotator 2)**: Completely isolated independent transcription and bounding-box delineation by a second rater.
- **Mode C (Adjudication)**: Side-by-side discrepancy adjudication, disagreement categorization, bounding-box consensus selection, and rationale documentation.

---

## 2. Privacy & Security Guarantees
- **Localhost Only**: Binds strictly to `127.0.0.1:5005`.
- **Zero External Calls**: No cloud APIs, external CDNs, or third-party web trackers. All rendering, computation, and storage happen locally on the investigator's workstation.
- **Decoupled Data Storage**: Raw page images are loaded via local filesystem paths (`AAE_DATA_ROOT`). Annotations are persisted to local JSON files (`annotations_store/`) which are excluded from public version control.
- **Audit Logging**: Every action, save, edit, and status change is immutably timestamped in `annotations_store/audit_log.jsonl`.

---

## 3. Launching the Tool

### Prerequisites
Activate the repository virtual environment:
```powershell
& "F:\Automatic Answer evaluation\.venv\Scripts\Activate.ps1"
```

Set the dataset root path (if different from default):
```powershell
$env:AAE_DATA_ROOT = "F:\Automatic answer evaluation dataset"
```

### Start Server
Run the Flask server:
```powershell
& "F:\Automatic Answer evaluation\.venv\Scripts\python.exe" research_v3/phase8_ground_truth/annotation_tool/app.py
```

Open a web browser and navigate to:
```
http://127.0.0.1:5005
```

---

## 4. Annotation Modes

### Mode A: Primary Blind Annotation
- Select **Mode A** in the navigation bar.
- Enter your Reviewer ID (format: `REV_A01`, `REV_A02`).
- Select a student booklet (`s1` through `s37`) and subquestion (`1a` through `4`).
- Inspect the high-resolution page scan.
- Draw or adjust the visual bounding box around the student's answer region.
- If the answer continues across pages or multiple regions, use the multi-segment bounding box feature.
- Transcribe the student's handwriting verbatim into the transcription textarea.
- Flag special attributes:
  - Contains Math / Formula
  - Contains Diagram / Drawing
  - Contains Crossed-out Text
  - Contains Overwrite / Correction
- Select Answer Status: `written`, `omitted_confirmed`, `illegible`, or `partially_legible`.
- Click **Save Annotation**.

### Mode B: Secondary Blind Annotation
- Select **Mode B** in the navigation bar.
- Enter your Reviewer ID (format: `REV_B02`).
- Perform the identical procedure independently. Mode B annotators cannot view Mode A transcriptions or bounding boxes.

### Mode C: Adjudication
- Select **Mode C** in the navigation bar.
- Enter your Adjudicator ID (format: `ADJ_LEAD01`).
- The interface displays:
  - Both Annotator A and Annotator B bounding boxes overlaid on the original page scan.
  - Word and Character Error Rate (CER / WER) between the two transcriptions.
  - Diff view highlighting spelling, token, punctuation, and omission discrepancies.
- Select the consensus bounding box or draw a corrected adjudicated box.
- Adjudicate the final ground-truth transcript.
- Select Disagreement Category:
  - `exact_agreement`
  - `minor_punctuation_or_casing`
  - `lexical_token_discrepancy`
  - `mathematical_symbol_discrepancy`
  - `status_discrepancy`
  - `unresolved`
- Provide explicit **Adjudication Rationale**.
- Flag whether the response is eligible for inclusion in the clean OCR benchmark (`inclusion_in_ocr_benchmark`).
- Click **Save Adjudication**.

---

## 5. Storage Directory Structure
All generated annotation data is saved locally to:
```
research_v3/phase8_ground_truth/annotations_store/
├── annotations_mode_a.json    # Mode A annotations
├── annotations_mode_b.json    # Mode B annotations
├── adjudications.json         # Mode C adjudicated ground-truth
└── audit_log.jsonl            # Immutable audit trail
```
This directory is gitignored to protect sensitive student handwriting and examination data.
