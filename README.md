# Automatic Answer Evaluation (v2)

Grade handwritten exam scripts in the teacher's own style, entirely on your computer
(PaddleOCR + Ollama + a QLoRA-fine-tuned local model). Flask app on `http://localhost:5000`.

## What changed in v2 - and why

1. **The old "OCR output" was not OCR.** In the version you ran, PaddleOCR was not installed, and the app
   silently fell back to *built-in placeholder text* (mitosis, supply curve, 5 questions x 20 marks, random
   marks). Your database even said `mode_used_paddleocr = False`. That is why it looked nothing like your exam.
   **v2 never fakes data any more**: no PaddleOCR -> a clear error. Fake data exists only if you start the app
   with `AAE_MODE=demo` (loud orange banner on every page).
2. **Exam first.** You upload the question paper + model answer key; they are read (exact text for digital PDFs,
   OCR for scans), you correct them once, and that exam drives everything else.
3. **The two folders you asked for**
   * `data/student_answers/<id>.json` - FOLDER 1, only the student's answers, keyed by question id
     (`"1a" ... "3c", "4"`).
   * `data/teacher_marks/<id>.json` - FOLDER 2, only the teacher's marks, keyed by question number
     (`"1".."4"`). These are your labels.
4. **Schema-guided parsing.** The answer sheet is split using the labels that exist in *your* paper (1a, 1b ... 4):
   many handwriting styles (`1a)`, `Q1 (a)`, `1.a.`, `Ans 4:`), recovery when a number is missed, no false
   question switches inside an answer.
5. **Red-ink separation.** On a corrected script the teacher's red marks sit on top of the student's writing.
   Each page is split by colour: answers are OCR'd with the red removed, marks are OCR'd from the red only.
6. **The AI selects the training papers** from folder 1 only (similarity to the model answer + completeness,
   spread from weakest to strongest). Their marks come from folder 2. Everything else is the test set and is
   graded after fine-tuning, then compared with its folder-2 marks.
7. **Verify screen** - scan on the left, parsed answers + marks on the right, so OCR slips are fixed
   before they become training labels.
8. **The fine-tuned model runs inside the app** (base + LoRA adapter, 4-bit). No GGUF conversion needed.
   The prompt used for training and for grading is the same code (`backend/prompts.py`).
9. **Honest evaluation**: compare *fine-tuned* vs *same base model without fine-tuning* vs an *Ollama model*.

## Quick start (Windows)

```bat
cd automatic-answer-evaluation
pip install -r requirements.txt
run_demo.bat        :: click through everything with synthetic students (no OCR / GPU needed)
run.bat             :: real mode
```
Open http://localhost:5000. In demo mode use **Exam setup -> "Use the sample LLM mid-sem paper + key"**
(your two PDFs are included in `examples/`).

## The workflow

| Step | Page | What happens |
|---|---|---|
| 1 | Exam setup | Upload question paper + model key -> parsed into questions, sub-questions, marks, model answers. Review, fix, **Approve**. |
| 2 | Answer sheets | Give the scans folder (`s1.pdf`, `s2.pdf` ... or sub-folders of photos). OCR writes folder 1 + folder 2. |
| 3 | Select & verify | The AI picks N training papers; open each, compare scan vs text vs marks, save. |
| 4 | Rubric & fine-tune | Build the rubric from the training papers, QLoRA fine-tune, grade the test papers with any engine. |
| 5 | Results | Predicted vs actual marks, error per question, totals, outliers (with "check" link to the scan), run comparison. |

## Setting up the real engines

1. **PDF + images**: `pip install pymupdf pillow numpy` (you have these).
2. **PaddleOCR**: install PaddlePaddle for your CUDA version, then `pip install paddleocr`. Both PaddleOCR 3.x and
   2.x are supported; the sidebar turns green when it is found. Set `AAE_OCR_DEVICE=cpu` to force CPU.
3. **Ollama**: install, then `ollama pull llama3.1:8b` (rubric / baseline) and optionally
   `ollama pull nomic-embed-text` (better selection).
4. **Fine-tuning**: install a CUDA build of PyTorch, then `pip install transformers peft bitsandbytes datasets accelerate`.
   Default base model `Qwen/Qwen2.5-3B-Instruct` (fits 4 GB VRAM in 4-bit). With an 8 GB card you may try a
   7B model: `set AAE_FINETUNE_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct` and `set AAE_GPU_VRAM_GB=8`.

## If OCR on your handwriting is not good enough

* Open a student on the **Verify** page: the warning chips tell you what the parser could not do
  (`blank_parts: 1c`, `marks_Q2: missing`, `marks_sum_mismatch`, `low_ocr_confidence`).
* **Marks**: if red-ink reading is unreliable (or the teacher used another colour) use the CSV import on the
  Answer sheets page (`roll_number,1,2,3,4`), or set the ink colour there.
* Page photos from `tools/handwritten_folders_to_pdf.py` (your script, unchanged) are fine as input.
* Label styles / roll-number format: `FULL_RE`, `Q_RE`, `SUB_RE`, `ROLL_RE` in `backend/sheet_parser.py`.
* PaddleOCR is a general OCR engine; messy cursive can be hard for it. If after tuning it is still poor, the next step is a
  handwriting-capable vision model behind the same row format (`ocr_engine.ocr_pages`) - the rest of the app would not change.

## Things to know (please read)

* **I could not run PaddleOCR, Ollama or the GPU parts in my environment.** The parsers, selection, pipeline, UI
  and all file formats are tested (`python tests/run_all.py`, 18 tests, using your real question paper and key); the
  PaddleOCR/Ollama/QLoRA calls are written against their documented APIs but first run on your machine is the real test.
  Send me the first `data/student_answers` / `data/teacher_marks` you get and a scan, and I will tune the parser.
* **Your key and paper disagree on Q3(c).** The paper asks *why layer normalisation is applied twice*; the key answers
  *how top-p differs from top-k*. The app flags it on the Exam setup page. Fix the paper or the key's answer before grading.
* **Small data**: 10 training papers x 4 questions = ~40 examples. That is enough to adapt marking *style*, not to
  guarantee accuracy; more verified training papers help most. Compare against the "no fine-tuning" run to see the real gain.
* The paper says *"No marks if all parts of a question are not answered together"*. By default the model is told which
  parts are blank and learns how strict your teacher actually is. `AAE_ENFORCE_ALL_PARTS=1` makes it a hard rule.
* Scanned pages and parsed JSON stay on this computer; nothing is sent anywhere.

## Settings (environment variables, see `config.py`)

`AAE_MODE` (real|demo), `AAE_DATA_DIR`, `AAE_ANSWERS_DIR`, `AAE_MARKS_DIR`, `AAE_OCR_DEVICE`, `AAE_TEACHER_INK`,
`AAE_SEPARATE_INK`, `AAE_ROLL_SOURCE` (filename|ocr), `AAE_OLLAMA_MODEL`, `AAE_SAMPLE_SIZE`,
`AAE_FINETUNE_BASE_MODEL`, `AAE_FINETUNE_EPOCHS`, `AAE_GPU_VRAM_GB`.

## Layout

```
app.py                 Flask routes
config.py              all settings
backend/exam_parser.py question paper + key -> exam        backend/sheet_parser.py OCR rows -> answers + marks
backend/ocr_engine.py  PaddleOCR 2.x/3.x wrapper           backend/image_utils.py  red-ink separation
backend/sampling.py    AI selection (label-free)           backend/prompts.py      one prompt for train + grade
backend/rubric_builder.py / grader.py / finetune.py / hf_infer.py / pipeline.py
examples/              your sample question paper + model answer
tools/                 your handwritten_folders_to_pdf.py + test_pdf_to_images.py (unchanged), optional GGUF export
tests/                 python tests/run_all.py
```
