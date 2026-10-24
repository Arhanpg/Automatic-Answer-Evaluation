"""
config.py -- all tunable settings in one place.

Every value can be overridden with an environment variable (AAE_*) so you
never have to edit code just to change a path or a model name.

Windows (cmd):         set AAE_MODE=demo && python app.py
Windows (PowerShell):  $env:AAE_MODE="demo"; python app.py
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AAE_DATA_DIR", os.path.join(BASE_DIR, "data"))

# ---- the exam definition (question paper + model answer key) ------------
EXAM_DIR = os.path.join(DATA_DIR, "exam")
EXAM_SOURCE_DIR = os.path.join(EXAM_DIR, "source")          # uploaded originals
EXAM_JSON = os.path.join(EXAM_DIR, "exam.json")              # approved schema
EXAM_DRAFT_JSON = os.path.join(EXAM_DIR, "exam_draft.json")  # parsed, not yet approved

# ---- the two output folders you asked for -------------------------------
# FOLDER 1: one JSON per student -> only the student's answers, by question id
STUDENT_ANSWERS_DIR = os.environ.get("AAE_ANSWERS_DIR", os.path.join(DATA_DIR, "student_answers"))
# FOLDER 2: one JSON per student -> only the marks the teacher gave, by question number
TEACHER_MARKS_DIR = os.environ.get("AAE_MARKS_DIR", os.path.join(DATA_DIR, "teacher_marks"))

PAGES_DIR = os.path.join(DATA_DIR, "pages")          # page images, for the verify screen
RUBRICS_DIR = os.path.join(DATA_DIR, "rubrics")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "db", "pipeline.sqlite3")

for _d in (EXAM_SOURCE_DIR, STUDENT_ANSWERS_DIR, TEACHER_MARKS_DIR, PAGES_DIR,
           RUBRICS_DIR, PREDICTIONS_DIR, MODELS_DIR, os.path.dirname(DB_PATH)):
    os.makedirs(_d, exist_ok=True)

# ---- runtime mode --------------------------------------------------------
# "real"  (default) -> real PaddleOCR / Ollama / GPU only. If something is
#                      missing you get a clear error, NEVER fake data.
# "demo"            -> synthetic students + a similarity-based stand-in grader,
#                      so you can click through the whole UI. Every page shows
#                      a loud DEMO banner.
RUNTIME_MODE = os.environ.get("AAE_MODE", "real").lower()
DEMO_MODE = RUNTIME_MODE == "demo"

ASSUMED_GPU_VRAM_GB = float(os.environ.get("AAE_GPU_VRAM_GB", "4"))   # set 8 if yours is 8GB

# ---- OCR -----------------------------------------------------------------
PADDLEOCR_LANG = os.environ.get("AAE_OCR_LANG", "en")
PADDLEOCR_DEVICE = os.environ.get("AAE_OCR_DEVICE", "gpu")   # "gpu" or "cpu"
OCR_DPI = int(os.environ.get("AAE_OCR_DPI", "200"))
# Digital PDFs (like your question paper / key) already contain exact text.
# Reading that is 100% accurate, so we use it and only OCR scanned pages.
PREFER_PDF_TEXT_LAYER = True

# Corrected scripts have the student's ink AND the teacher's red marks on
# the same page. We split them into two images by colour: the student's
# answers are OCR'd with the red removed, the marks are OCR'd from red only.
SEPARATE_TEACHER_INK = os.environ.get("AAE_SEPARATE_INK", "1") == "1"
TEACHER_INK = os.environ.get("AAE_TEACHER_INK", "red")       # "red" or "green"
MIN_TEACHER_INK_FRACTION = 0.0004   # below this the page is treated as having no marks

# "filename" -> student id = file / sub-folder name (s1.pdf -> s1)   [default]
# "ocr"      -> try to read "Roll No" from page 1, fall back to the filename
ROLL_SOURCE = os.environ.get("AAE_ROLL_SOURCE", "filename")

# ---- Ollama --------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("AAE_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.environ.get("AAE_EMBED_MODEL", "nomic-embed-text")

# ---- selection / grading -------------------------------------------------
DEFAULT_SAMPLE_SIZE = int(os.environ.get("AAE_SAMPLE_SIZE", "10"))
GRADE_STEP = 0.5                      # marks are rounded to the nearest 0.5
MAX_PART_CHARS = 700                  # per-answer cap inside prompts
# Your paper says "No marks if all parts of a question are not answered".
# False  -> the model is TOLD which parts are blank and learns how strict
#           your teacher really is from the labelled papers (recommended).
# True   -> hard rule: any blank part => 0 marks for that question.
ENFORCE_ALL_PARTS_RULE = os.environ.get("AAE_ENFORCE_ALL_PARTS", "0") == "1"

# ---- fine-tuning (QLoRA) -------------------------------------------------
FINETUNE_BASE_MODEL = os.environ.get("AAE_FINETUNE_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
FINETUNE_EPOCHS = int(os.environ.get("AAE_FINETUNE_EPOCHS", "4"))
FINETUNE_LR = 2e-4
FINETUNE_MAX_TOKENS = 1024
USE_4BIT_QUANTIZATION = True
ADAPTER_DIR = os.path.join(MODELS_DIR, "lora_adapter")

SECRET_KEY = os.environ.get("AAE_SECRET_KEY", "dev-key-change-me")
