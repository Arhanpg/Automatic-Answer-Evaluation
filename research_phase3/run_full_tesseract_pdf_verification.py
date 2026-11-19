"""Extract and cross-check Tesseract text for every rendered answer-sheet page."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import cv2
import pytesseract
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_phase3"
IMAGE_DIR = OUT / "answer_region_images"
GOLD = ROOT / "data" / "ground_truth" / "verified_transcriptions.json"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+\-=<>/().\[\]{}_^√ ×*%:,\"]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not shutil.which("tesseract"):
        raise RuntimeError("Tesseract executable is unavailable")

    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    page_results = {}
    page_rows = []
    images = sorted(IMAGE_DIR.glob("*_page_[0-9][0-9][0-9].png"))

    for index, image_path in enumerate(images, 1):
        match = re.match(r"(.+)_page_(\d{3})\.png$", image_path.name)
        if not match:
            continue
        student_id, page_number = match.group(1), int(match.group(2))
        image = cv2.imread(str(image_path))
        started = time.perf_counter()
        try:
            text = pytesseract.image_to_string(image, config="--psm 6").strip()
            status = "executed"
        except Exception as exc:
            text = ""
            status = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        key = f"{student_id}|{page_number}"
        page_results[key] = {
            "student_id": student_id,
            "page_number": page_number,
            "execution_status": status,
            "latency_seconds": round(elapsed, 4),
            "character_count": len(text),
            "word_count": len(text.split()),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
        page_rows.append({
            "student_id": student_id,
            "page_number": page_number,
            "execution_status": status,
            "latency_seconds": round(elapsed, 4),
            "character_count": len(text),
            "word_count": len(text.split()),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "page_image_hash": sha256(image_path),
        })

    (OUT / "full_tesseract_page_ocr.json").write_text(
        json.dumps(page_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (OUT / "full_tesseract_page_ocr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(page_rows[0]))
        writer.writeheader()
        writer.writerows(page_rows)

    review_rows = []
    for student_id in sorted(gold, key=lambda value: int(value[1:])):
        answers = gold[student_id]["answers"]
        for subquestion_id, reference in answers.items():
            candidates = []
            reference_norm = normalize(reference)
            for key, result in page_results.items():
                if result["student_id"] != student_id:
                    continue
                page_norm = normalize(result["text"])
                whole = fuzz.token_set_ratio(reference_norm, page_norm)
                token = fuzz.partial_ratio(reference_norm, page_norm)
                candidates.append((max(whole, token), result["page_number"], whole, token))
            candidates.sort(reverse=True)
            if not candidates:
                status = "source_page_missing"
                page_number = ""
                whole = token = 0.0
            else:
                score, page_number, whole, token = candidates[0]
                if score >= 55:
                    status = "strong_page_reference_match_pending_crop_review"
                elif score >= 35:
                    status = "weak_page_reference_match_pending_crop_review"
                else:
                    status = "unresolved_manual_review_required"

            review_rows.append({
                "student_id": student_id,
                "subquestion_id": subquestion_id,
                "reference_transcription": reference,
                "ocr_engine": "Tesseract 5",
                "candidate_page": page_number,
                "token_set_similarity": round(whole, 2),
                "partial_similarity": round(token, 2),
                "review_status": status,
                "ocr_output_available": page_number != "",
            })

    fields = list(review_rows[0])
    with (OUT / "ocr_reference_page_review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    print(json.dumps({
        "pages": len(page_rows),
        "subquestions": len(review_rows),
        "strong_matches": sum(r["review_status"] == "strong_page_reference_match_pending_crop_review" for r in review_rows),
        "weak_matches": sum(r["review_status"] == "weak_page_reference_match_pending_crop_review" for r in review_rows),
        "unresolved": sum(r["review_status"] == "unresolved_manual_review_required" for r in review_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
