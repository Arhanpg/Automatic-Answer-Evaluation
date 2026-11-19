"""Generate non-destructive provenance and reproducibility audit artifacts.

The script records metadata only. It never copies or modifies examination scans,
existing experiment outputs, labels, or predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("ANSWER_EVAL_DATASET", r"F:\Automatic answer evaluation dataset"))
AUDIT = ROOT / "research_audit"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with (AUDIT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_inventory() -> None:
    rows = []
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"], text=False
    ).decode("utf-8").split("\0")
    for relative in sorted(path for path in tracked if path and not path.startswith("research_audit/")):
        path = ROOT / relative
        if not path.is_file():
            continue
        rel = relative.replace("\\", "/")
        suffix = path.suffix.lower()
        if rel.startswith(("data/", "outputs/", "experiments/")):
            kind = "dataset/output/report"
            evidence = "generated_or_recorded; inspect per-file provenance"
        elif suffix in {".py", ".bat"}:
            kind = "source code"
            evidence = "implementation; execution not implied"
        elif suffix in {".md", ".txt"}:
            kind = "documentation"
            evidence = "claim or instructions; not execution proof"
        elif suffix in {".pdf", ".jpg", ".jpeg", ".png"}:
            kind = "example image/document"
            evidence = "stored artifact; visual validation not implied"
        else:
            kind = "configuration/supporting file"
            evidence = "supporting artifact"
        rows.append(
            {
                "path": rel,
                "type": kind,
                "purpose": "Repository artifact; see source path and adjacent report.",
                "provenance": "Tracked in the repository at audit time.",
                "associated_experiment": re.search(r"(M\d+)", rel, re.I).group(1)
                if re.search(r"(M\d+)", rel, re.I)
                else "",
                "evidence_status": evidence,
                "research_relevance": "Potential evidence; requires protocol and execution verification.",
            }
        )
    write_csv(
        "file_inventory.csv",
        rows,
        [
            "path",
            "type",
            "purpose",
            "provenance",
            "associated_experiment",
            "evidence_status",
            "research_relevance",
        ],
    )


def dataset_artifacts() -> dict[str, Any]:
    manifest = read_json(ROOT / "outputs/dataset_audit/student_split_manifest.json")
    gold = read_json(ROOT / "data/ground_truth/verified_transcriptions.json")
    answer_ids = {p.stem for p in (ROOT / "data/student_answers").glob("s*.json")}
    mark_ids = {p.stem for p in (ROOT / "data/teacher_marks").glob("s*.json")}
    pdf_dir = DATASET_ROOT / "All PDF's"
    split_by_id = {
        student_id: split
        for split, data in manifest["splits"].items()
        for student_id in data["student_ids"]
    }

    rows = []
    for student_id in sorted(split_by_id, key=lambda value: int(value[1:])):
        marks = gold.get(student_id, {}).get("examiner_marks", {})
        rows.append(
            {
                "student_id": student_id,
                "split": split_by_id[student_id],
                "booklet_id": student_id,
                "question_count_expected": 4,
                "student_answer_file": f"data/student_answers/{student_id}.json",
                "teacher_marks_file": f"data/teacher_marks/{student_id}.json",
                "verified_transcription_present": student_id in gold,
                "answer_file_present": student_id in answer_ids,
                "marks_file_present": student_id in mark_ids,
                "source_pdf_present": (pdf_dir / f"{student_id}.pdf").exists(),
                "source_pdf_path": str((pdf_dir / f"{student_id}.pdf").resolve())
                if (pdf_dir / f"{student_id}.pdf").exists()
                else "",
                "source_pdf_pages": read_json(ROOT / "outputs/dataset_audit/corpus_file_audit.json")
                .get("scripts", {})
                .get(student_id, {})
                .get("pages", ""),
                "ground_truth_total_mark": marks.get("total", ""),
            }
        )
    write_csv(
        "dataset_manifest.csv",
        rows,
        list(rows[0]),
    )

    provenance_rows = [
        {
            "artifact": "verified_transcriptions.json",
            "scope": "37 student records; answer text and examiner marks",
            "path": "data/ground_truth/verified_transcriptions.json",
            "provenance_found": "File is labelled verified, but annotation workflow or independent grader record is not stored.",
            "evidence_status": "repository-labelled; independent human validation unverified",
            "safe_research_use": "Reference input only after annotation provenance is documented.",
        },
        {
            "artifact": "student_answers",
            "scope": "37 student records",
            "path": "data/student_answers/*.json",
            "provenance_found": "Structured answer exports; source transformation and correction history not stored per record.",
            "evidence_status": "recorded/generated; provenance incomplete",
            "safe_research_use": "Use only with explicit source-to-record mapping.",
        },
        {
            "artifact": "teacher_marks",
            "scope": "37 student records",
            "path": "data/teacher_marks/*.json",
            "provenance_found": "Structured mark exports; grader identity, rubric adjudication, and inter-rater evidence absent.",
            "evidence_status": "recorded; independent grading validation unverified",
            "safe_research_use": "Single-rater labels unless additional graders are collected.",
        },
        {
            "artifact": "training_set.jsonl",
            "scope": "training examples",
            "path": "data/models/training_set.jsonl",
            "provenance_found": "Tracked file contains zero lines at audit time.",
            "evidence_status": "empty",
            "safe_research_use": "Cannot support a training-data claim.",
        },
    ]
    write_csv("ground_truth_provenance.csv", provenance_rows, list(provenance_rows[0]))
    return manifest


def visual_and_ocr_artifacts() -> None:
    audit = read_json(ROOT / "outputs/dataset_audit/corpus_file_audit.json")
    preprocessing = ROOT / "static/images/preprocessing_samples"
    rows = []
    for student_id, details in audit.get("scripts", {}).items():
        rows.append(
            {
                "student_id": student_id,
                "question_id": "not recorded",
                "image_condition": "source PDF metadata only",
                "source_path": details.get("pdf_path", ""),
                "visual_inspection_status": "not independently performed by this audit",
                "mathematical_symbol_review": "not assessed",
                "red_ink_review": "not assessed",
                "comments": "Page count is recorded; no claim about readable handwriting or transcription correctness.",
            }
        )
    write_csv("dataset_visual_audit.csv", rows, list(rows[0]))

    live_csv = ROOT / "outputs/ocr_metrics/live_per_script_metrics.csv"
    live_rows = []
    with live_csv.open(encoding="utf-8", newline="") as handle:
        live_rows.extend(csv.DictReader(handle))
    for row in live_rows:
        row.update(
            {
                "evidence_status": "recorded live telemetry file; execution environment and command are not bundled",
                "modality": "raw/live per-script",
            }
        )
    write_csv("ocr_sample_level_results.csv", live_rows, list(live_rows[0]) if live_rows else ["student_id"])

    ocr_rows = [
        {
            "track": "raw OCR",
            "engine": "PaddleOCR",
            "result_path": "outputs/independent_benchmarks/paddle_ocr_benchmark.json",
            "status": "reported as scripted/calibrated in M2 report",
            "metrics_comparable": "no",
        },
        {
            "track": "preprocessed OCR",
            "engine": "Tesseract",
            "result_path": "outputs/independent_benchmarks/tesseract_ocr_benchmark.json",
            "status": "reported as scripted/calibrated in M2 report",
            "metrics_comparable": "no",
        },
        {
            "track": "vision transcription",
            "engine": "TrOCR/VLM label",
            "result_path": "outputs/independent_benchmarks/vlm_benchmark.json",
            "status": "reported as scripted/calibrated; actual model execution unverified",
            "metrics_comparable": "no",
        },
        {
            "track": "verified transcription",
            "engine": "human/reference label",
            "result_path": "data/ground_truth/verified_transcriptions.json",
            "status": "repository-labelled; annotation workflow unverified",
            "metrics_comparable": "reference only",
        },
    ]
    write_csv("ocr_track_comparison.csv", ocr_rows, list(ocr_rows[0]))
    (AUDIT / "visual_comparisons").mkdir(exist_ok=True)
    (AUDIT / "visual_comparisons/README.md").write_text(
        "# Visual comparison status\n\n"
        "The audit records source-image metadata and existing preprocessing samples, "
        "but does not claim independent visual inspection or human transcription validation. "
        "No examination scans are copied into this directory. Existing samples are under "
        "`static/images/preprocessing_samples/`.\n",
        encoding="utf-8",
    )


def model_artifacts() -> None:
    leaderboard = ROOT / "outputs/grading_performance/model_leaderboard.csv"
    rows = []
    with leaderboard.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["evidence_status"] = "generated leaderboard; modality-specific live execution not established"
            row["input_modality"] = "not recorded in leaderboard"
            rows.append(row)
    write_csv("grading_results_by_modality.csv", rows, list(rows[0]))
    matrix = []
    for row in rows:
        matrix.append(
            {
                "model": row["model"],
                "paradigm": row["paradigm"],
                "implementation_location": "backend/src implementation exists or model is named in generated output",
                "weights_or_external_service_verified": "no",
                "training_execution_verified": "no",
                "locked_test_predictions_verified": "no",
                "synthetic_or_scripted_inputs_possible": "yes",
                "reproducibility_status": "incomplete; preserve modality and execution log before publication claim",
            }
        )
    write_csv("modality_model_matrix.csv", rows, list(rows[0]))
    write_csv("model_reproducibility_matrix.csv", matrix, list(matrix[0]))
    write_csv("grading_sample_level_results.csv", [], ["student_id", "question_id", "model", "input_modality", "prediction", "ground_truth", "evidence_status"])


def reports(manifest: dict[str, Any]) -> None:
    split_hash = manifest["split_sha256"]
    (AUDIT / "01_dataset_and_ground_truth_audit.md").write_text(
        f"""# Dataset and ground-truth audit

## Verified observations

- The repository manifest contains 37 booklet IDs and 148 question responses:
  25 train, 6 validation, and 6 locked-test booklets.
- The stored split hash is `{split_hash}`. It is reproducible from the manifest
  serialization used by `backend/split_validation.py`; the three sets are disjoint.
- The external source directory was present during this audit and the existing
  corpus audit records 37 PDFs and 195 pages. This audit stores metadata only and
  does not copy scans or expose student names/registration numbers.
- `data/ground_truth/verified_transcriptions.json`, student answers, and teacher
  marks contain 37 IDs. The repository does not contain an independent annotation
  protocol, second-grader labels, or inter-rater agreement.
- `data/models/training_set.jsonl` is empty at audit time.

## Limitations and leakage controls

The unit of independence is the student booklet, not the 148 question rows.
The six locked-test students therefore provide only six independent test groups;
question-level metrics may be useful diagnostics but cannot be interpreted as
148 independent student observations. The stored GroupKFold configuration is
student-grouped, but every reported model result still requires a run log linking
inputs, code revision, seed, and predictions.

The source code in `backend/classical_ml.py` contains a deterministic synthetic
fallback for IDs absent from the gold records. That behavior must be disclosed
and disabled for a real-data claim. Existing reports also describe several
benchmark values as generated or simulated; they remain separate from this audit.

## Required follow-up before a paper

Document the transcription/marking workflow, obtain independent grading or
adjudication, preserve source-to-answer mappings, run deduplication and
near-duplicate checks at student level, and collect external students or a new
exam for validation. Do not tune on the locked test set.
""",
        encoding="utf-8",
    )
    (AUDIT / "dataset_integrity_report.md").write_text(
        """# Dataset integrity report

| Check | Result | Evidence |
|---|---|---|
| Expected student IDs | PASS: s1-s37 present in source audit | `outputs/dataset_audit/corpus_file_audit.json` |
| Expected response count | PASS: 37 x 4 = 148 in manifest | `outputs/dataset_audit/student_split_manifest.json` |
| Split overlap | PASS: no train/validation/test ID overlap | `backend/split_validation.py` and manifest |
| Source page count | RECORDED: 195 pages | corpus audit JSON |
| Independent label provenance | UNVERIFIED | no annotation log or second grader stored |
| Duplicate/near-duplicate answer audit | NOT ESTABLISHED | no canonical deduplication report found |
| Training JSONL | FAIL for training claim: empty | `data/models/training_set.jsonl` |
| Test independence | LIMITED | only six student groups in locked test |

This report is an evidence inventory, not a claim that the dataset is suitable
for publication without additional validation.
""",
        encoding="utf-8",
    )
    (AUDIT / "02_preprocessing_and_ocr_audit.md").write_text(
        """# Preprocessing and OCR audit

The implementation in `backend/independent_pipeline.py` contains the reported
four stages: HSV red-ink isolation, grayscale/grid suppression/deskew,
Fast Non-Local Means denoising, and Sauvola binarization. The configured values
include `h=10`, template/search windows 7/21, and Sauvola window 25, `k=0.2`.

The repository contains representative preprocessing images under
`static/images/preprocessing_samples/`, but this audit does not claim a complete
independent visual review of all source pages or human verification of symbols.

The tracked M2 report explicitly says the independent PaddleOCR, Tesseract, and
VLM tables are scripted/calibrated from verified text. The live CSV contains only
two rows and very high raw CER/WER. Those tracks are therefore kept separate in
`ocr_track_comparison.csv` and must not be combined into a preprocessing
improvement claim. Mathematical equivalence is not established by textual CER/WER.
""",
        encoding="utf-8",
    )
    (AUDIT / "03_model_execution_audit.md").write_text(
        """# Model execution audit

The repository contains implementations for classical ML, from-scratch
deep-learning, symbolic, retrieval, LLM, and fine-tuning paths. The tracked
leaderboard does not include an input-modality identifier or a per-sample
prediction/execution manifest. The model matrix consequently marks execution,
weights, and locked-test provenance as incomplete rather than inferring them
from a CSV.

The model leaderboard is retained in `grading_results_by_modality.csv` with an
explicit generated-artifact status. Results from raw OCR, preprocessed OCR,
vision transcription, and verified reference text must be regenerated under a
single fixed protocol before any cross-modality comparison.
""",
        encoding="utf-8",
    )
    (AUDIT / "04_results_and_evidence_audit.md").write_text(
        """# Results and evidence audit

The repository documentation already distinguishes recorded implementation
outputs from simulated/calibrated benchmark artifacts. This audit confirms
that distinction and records the following high-risk inconsistencies:

- The hardware registry contains historical entries with both zero and nonzero
  VRAM; current evidence does not establish a live GPU run.
- The registry contains contradictory historical classical-model metrics.
- The M10 artifact reports apparent test coverage but the report states zero
  calibration records, so it is not a valid calibrated guarantee.
- The independent OCR benchmark names external engines while its report says
  text variations were scripted from verified transcriptions.

No existing result or source dataset was overwritten. The CSVs in this directory
are evidence-index artifacts and are not replacement benchmark results.
""",
        encoding="utf-8",
    )


def main() -> None:
    AUDIT.mkdir(exist_ok=True)
    manifest = dataset_artifacts()
    source_inventory()
    visual_and_ocr_artifacts()
    model_artifacts()
    reports(manifest)
    print(f"Generated audit artifacts in {AUDIT}")


if __name__ == "__main__":
    main()
