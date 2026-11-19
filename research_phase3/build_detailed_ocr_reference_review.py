"""Merge full-page OCR text with every subquestion reference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_phase3"


def main() -> None:
    page_ocr = json.loads((OUT / "full_tesseract_page_ocr.json").read_text(encoding="utf-8"))
    review = {}
    with (OUT / "ocr_reference_page_review.csv").open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            review[(row["student_id"], row["subquestion_id"])] = row

    detailed = []
    with (OUT / "answer_region_manifest.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        key = (row["student_id"], row["subquestion_id"])
        match = review.get(key, {})
        candidate_page = match.get("candidate_page", "")
        page_text = ""
        if candidate_page:
            page_text = page_ocr.get(
                f"{row['student_id']}|{int(candidate_page)}",
                {},
            ).get("text", "")

        if match.get("review_status") == "strong_page_reference_match_pending_crop_review":
            annotation_status = "ocr_actual_vs_reference_strong_candidate"
        elif match.get("review_status") == "weak_page_reference_match_pending_crop_review":
            annotation_status = "ocr_actual_vs_reference_weak_candidate"
        elif match.get("review_status") == "unresolved_manual_review_required":
            annotation_status = "ocr_actual_vs_reference_unresolved"
        else:
            annotation_status = "ocr_actual_vs_reference_no_source_page"

        detailed.append({
            **row,
            "ocr_engine": "Tesseract 5",
            "ocr_extracted_page_text": page_text,
            "ocr_candidate_page": candidate_page,
            "reference_vs_actual_similarity": match.get("token_set_similarity", ""),
            "reference_partial_similarity": match.get("partial_similarity", ""),
            "ocr_actual_vs_reference_review_status": annotation_status,
        })

    fields = list(detailed[0])
    with (OUT / "detailed_ocr_reference_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detailed)

    queue = []
    for row in detailed:
        queue.append({
            "student_id": row["student_id"],
            "review_type": "actual_pdf_ocr_vs_reference",
            "page_number": row["ocr_candidate_page"],
            "question_id": row["question_id"],
            "subquestion_id": row["subquestion_id"],
            "reference_vs_actual_similarity": row["reference_vs_actual_similarity"],
            "partial_similarity": row["reference_partial_similarity"],
            "annotation_status": row["ocr_actual_vs_reference_review_status"],
            "reason": "Full-page OCR text compared with repository reference; exact answer crop not yet verified",
        })

    queue_fields = list(queue[0])
    with (OUT / "mapping_review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=queue_fields)
        writer.writeheader()
        writer.writerows(queue)

    print({
        "detailed_records": len(detailed),
        "strong": sum(r["ocr_actual_vs_reference_review_status"] == "ocr_actual_vs_reference_strong_candidate" for r in detailed),
        "weak": sum(r["ocr_actual_vs_reference_review_status"] == "ocr_actual_vs_reference_weak_candidate" for r in detailed),
        "unresolved": sum(r["ocr_actual_vs_reference_review_status"] == "ocr_actual_vs_reference_unresolved" for r in detailed),
    })


if __name__ == "__main__":
    main()
