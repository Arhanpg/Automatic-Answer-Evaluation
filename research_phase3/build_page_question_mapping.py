"""Create a non-destructive page/question mapping review artifact.

This script does not infer answer boundaries. It records the actual PDF/page,
PDF page text, OCR headings found by Tesseract, and every verified reference
question/subquestion. Rows requiring visual adjudication remain unresolved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("ANSWER_EVAL_DATASET", r"F:\Automatic answer evaluation dataset"))
PDF_DIR = DATASET / "All PDF's"
OUT = ROOT / "research_phase3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_id(value: str) -> str:
    return str(value).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_images = OUT / "answer_region_images"
    out_images.mkdir(parents=True, exist_ok=True)

    gold = json.loads((ROOT / "data" / "ground_truth" / "verified_transcriptions.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "outputs" / "dataset_audit" / "student_split_manifest.json").read_text(encoding="utf-8"))
    splits = {
        sid: split_name
        for split_name, split_data in manifest["splits"].items()
        for sid in split_data["student_ids"]
    }

    rows = []
    queue = []
    page_rows = []

    for student_id in sorted(gold, key=lambda value: int(value[1:])):
        pdf = PDF_DIR / f"{student_id}.pdf"
        record = gold[student_id]
        if not pdf.exists():
            queue.append({
                "student_id": student_id,
                "review_type": "missing_source_pdf",
                "page_number": "",
                "question_id": "",
                "reason": "source PDF not found",
            })
            continue

        doc = fitz.open(pdf)
        for page_number in range(doc.page_count):
            page = doc[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            page_image = out_images / f"{student_id}_page_{page_number + 1:03d}.png"
            pix.save(page_image)
            page_text = page.get_text("text").strip()
            try:
                ocr_text = pytesseract.image_to_string(
                    Image.open(page_image),
                    config="--psm 6",
                ).strip()
            except Exception as exc:
                ocr_text = ""
                page_text += f"\nOCR_ERROR={type(exc).__name__}: {exc}"

            page_rows.append({
                "student_id": student_id,
                "split": splits.get(student_id, ""),
                "page_number": page_number + 1,
                "source_pdf": str(pdf.resolve()),
                "source_pdf_sha256": sha256(pdf),
                "page_image": str(page_image.resolve()),
                "page_image_sha256": hashlib.sha256(page_image.read_bytes()).hexdigest(),
                "pdf_text_present": bool(page_text.strip()),
                "ocr_heading_review_status": "pending_manual_visual_review",
                "pdf_text_preview": page_text[:2000].replace("\n", " "),
                "ocr_text_preview": ocr_text[:2000].replace("\n", " "),
            })

        marks = record.get("examiner_marks", {})
        sub_marks = record.get("sub_marks", {})
        answers = record.get("answers", {})
        for subquestion_id, reference in answers.items():
            question_id = subquestion_id.split("a", 1)[0].split("b", 1)[0].split("c", 1)[0].split("d", 1)[0]
            if question_id not in {"1", "2", "3", "4"}:
                question_id = subquestion_id
            rows.append({
                "student_id": student_id,
                "split": splits.get(student_id, ""),
                "page_number": "",
                "question_id": question_id,
                "subquestion_id": subquestion_id,
                "bounding_box": "",
                "source_pdf": str(pdf.resolve()),
                "source_pdf_sha256": sha256(pdf),
                "answer_crop_hash": "",
                "reference_transcription": reference,
                "reference_provenance": "repository-verified_transcriptions.json; independent human annotation workflow not recorded",
                "examiner_mark": sub_marks.get(subquestion_id, marks.get(question_id, "")),
                "maximum_mark": "",
                "annotation_status": "unmapped_pending_manual_review",
            })
            queue.append({
                "student_id": student_id,
                "review_type": "answer_region_mapping",
                "page_number": "",
                "question_id": question_id,
                "subquestion_id": subquestion_id,
                "reason": "PDF page and exact bounding box not manually verified",
            })

    fields = [
        "student_id", "split", "page_number", "question_id", "subquestion_id",
        "bounding_box", "source_pdf", "source_pdf_sha256", "answer_crop_hash",
        "reference_transcription", "reference_provenance", "examiner_mark",
        "maximum_mark", "annotation_status",
    ]
    with (OUT / "answer_region_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    queue_fields = ["student_id", "review_type", "page_number", "question_id", "subquestion_id", "reason"]
    with (OUT / "mapping_review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=queue_fields)
        writer.writeheader()
        writer.writerows(queue)

    page_fields = list(page_rows[0]) if page_rows else [
        "student_id", "split", "page_number", "source_pdf", "source_pdf_sha256",
        "page_image", "page_image_sha256", "pdf_text_present",
        "ocr_heading_review_status", "pdf_text_preview", "ocr_text_preview",
    ]
    with (OUT / "page_review_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=page_fields)
        writer.writeheader()
        writer.writerows(page_rows)

    print(json.dumps({
        "students": len(gold),
        "pages_rendered": len(page_rows),
        "reference_regions": len(rows),
        "manual_review_rows": len(queue),
    }, indent=2))


if __name__ == "__main__":
    main()
