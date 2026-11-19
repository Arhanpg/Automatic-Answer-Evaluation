"""Cross-check PDF page images with repository references and OCR outputs.

This performs automated visual-text verification. It does not claim an independent
human transcription; uncertain rows remain in the review queue.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import cv2
import pytesseract
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "research_phase3"
DATASET = ROOT / "data" / "ground_truth" / "verified_transcriptions.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+\-=<>/().\[\]{}_^√ ×*%: ,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tesseract_page(image_path: Path) -> tuple[str, list[dict], dict]:
    image = cv2.imread(str(image_path))
    data = pytesseract.image_to_data(
        image,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    text = pytesseract.image_to_string(image, config="--psm 6").strip()
    lines = {}
    for i, word in enumerate(data["text"]):
        word = str(word).strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        item = lines.setdefault(key, {"words": [], "x0": 10**9, "y0": 10**9, "x1": 0, "y1": 0})
        item["words"].append(word)
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        item["x0"] = min(item["x0"], x)
        item["y0"] = min(item["y0"], y)
        item["x1"] = max(item["x1"], x + w)
        item["y1"] = max(item["y1"], y + h)

    line_rows = []
    for item in lines.values():
        line_rows.append({
            "text": " ".join(item["words"]),
            "x0": item["x0"],
            "y0": item["y0"],
            "x1": item["x1"],
            "y1": item["y1"],
        })
    return text, line_rows, {"width": image.shape[1], "height": image.shape[0]}


def label_variants(subquestion_id: str) -> list[str]:
    sid = subquestion_id.lower()
    compact = sid.replace(".", "")
    if re.fullmatch(r"\d+[a-d]", compact):
        number, letter = compact[:-1], compact[-1]
        return [
            compact,
            f"{number}{letter})",
            f"{number} {letter}",
            f"{number} {letter})",
            f"({letter})",
            f"{letter})",
        ]
    if compact == "3":
        return ["3)", "3.", "question 3", "q3"]
    if compact == "4":
        return ["4)", "4.", "question 4", "q4"]
    return [compact]


def find_label(lines: list[dict], subquestion_id: str) -> dict | None:
    variants = label_variants(subquestion_id)
    compact = subquestion_id.lower().replace(".", "")
    scored = []
    for line in lines:
        line_text = normalize(line["text"])
        if not line_text:
            continue
        if compact in line_text or any(v in line_text for v in variants):
            if re.search(rf"\b{re.escape(compact)}\b", line_text) or compact in line_text:
                scored.append((100, line))
        else:
            ratio = max(fuzz.partial_ratio(v, line_text) for v in variants)
            if ratio >= 82:
                scored.append((ratio, line))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def crop_bbox(lines: list[dict], shape: dict, subquestion_id: str) -> tuple[str, tuple[int, int, int, int]]:
    label = find_label(lines, subquestion_id)
    if label is None:
        return "page_unresolved", (0, 0, shape["width"], shape["height"])

    if re.fullmatch(r"[1-4][a-d]", subquestion_id.lower()):
        label_text = subquestion_id.lower()
    else:
        label_text = subquestion_id.lower()

    y0 = max(0, label["y0"] - 10)
    y1 = shape["height"]
    # Stop at the next subquestion/question label, if a later OCR line is plausible.
    later = [line for line in lines if line["y0"] > label["y1"]]
    for line in later:
        next_text = normalize(line["text"])
        if re.search(r"\b[1-4]\s*[a-d]\b|\b[1-4]\s*[).]\b", next_text):
            y1 = min(y1, max(line["y0"] - 5, y0))
            break

    x0 = max(0, label["x0"] - 20)
    x1 = shape["width"]
    if y1 <= y0:
        y1 = shape["height"]
    return "label_detected", (x0, y0, x1, y1)


def main() -> None:
    gold = json.loads(DATASET.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "outputs" / "dataset_audit" / "student_split_manifest.json").read_text(encoding="utf-8"))
    splits = {
        sid: split_name
        for split_name, data in manifest["splits"].items()
        for sid in data["student_ids"]
    }

    page_images = {}
    for image in (OUT / "answer_region_images").glob("*_page_*.png"):
        match = re.match(r"(.+)_page_(\d{3})\.png$", image.name)
        if match:
            page_images[(match.group(1), int(match.group(2)))] = image

    page_ocr = {}
    for key, image in page_images.items():
        page_text, lines, shape = tesseract_page(image)
        page_ocr[key] = {"text": page_text, "lines": lines, "shape": shape}

    result_rows = []
    queue_rows = []
    for student_id in sorted(gold, key=lambda value: int(value[1:])):
        record = gold[student_id]
        answers = record["answers"]
        sub_marks = record.get("sub_marks", {})
        marks = record.get("examiner_marks", {})
        student_pages = sorted(
            [(number, values) for (sid, number), values in page_ocr.items() if sid == student_id],
            key=lambda item: item[0],
        )

        for subquestion_id, reference in answers.items():
            question_id = re.match(r"\d+", subquestion_id).group(0)
            page_scores = []
            ref_norm = normalize(reference)
            for page_number, values in student_pages:
                page_norm = normalize(values["text"])
                score = fuzz.token_set_ratio(ref_norm, page_norm)
                page_scores.append((score, page_number, values))
            page_scores.sort(key=lambda item: item[0], reverse=True)

            top_score, top_page, top_values = page_scores[0]
            label_status, bbox = crop_bbox(top_values["lines"], top_values["shape"], subquestion_id)

            crop_image = cv2.imread(str(page_images[(student_id, top_page)]))
            x0, y0, x1, y1 = bbox
            crop = crop_image[y0:y1, x0:x1]
            crop_dir = OUT / "answer_region_images" / "verified_crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_path = crop_dir / f"{student_id}_{subquestion_id}_p{top_page:03d}.png"
            cv2.imwrite(str(crop_path), crop)

            crop_text = pytesseract.image_to_string(crop, config="--psm 6").strip()
            cross_score = fuzz.token_set_ratio(normalize(crop_text), ref_norm)
            whole_score = fuzz.token_set_ratio(normalize(top_values["text"]), ref_norm)

            if label_status == "label_detected" and top_score >= 35 and cross_score >= 25:
                status = "automatically_cross_verified_pending_independent_review"
            elif top_score >= 55:
                status = "page_candidate_cross_verified_pending_independent_review"
            else:
                status = "unresolved_manual_review_required"

            source_pdf = page_images[(student_id, top_page)].resolve()
            data_dir = OUT / "answer_region_images"
            manifest_path = data_dir / f"{student_id}_page_{top_page:03d}.png"
            result_rows.append({
                "student_id": student_id,
                "split": splits.get(student_id, ""),
                "page_number": top_page,
                "question_id": question_id,
                "subquestion_id": subquestion_id,
                "bounding_box": f"[{x0},{y0},{x1},{y1}]",
                "source_pdf": f"F:\\Automatic answer evaluation dataset\\All PDF's\\{student_id}.pdf",
                "source_pdf_sha256": sha256(Path(f"F:\\Automatic answer evaluation dataset\\All PDF's\\{student_id}.pdf")),
                "answer_crop": str(crop_path.resolve()),
                "answer_crop_hash": sha256(crop_path),
                "reference_transcription": reference,
                "reference_provenance": "repository verified_transcriptions.json; not independently adjudicated",
                "ocr_extracted_text": crop_text,
                "ocr_engine": "Tesseract 5",
                "ocr_method": "page-image -> PSM 6 crop OCR",
                "page_reference_similarity": round(top_score, 2),
                "crop_reference_similarity": round(cross_score, 2),
                "whole_page_similarity": round(whole_score, 2),
                "examiner_mark": sub_marks.get(subquestion_id, marks.get(question_id, "")),
                "maximum_mark": "",
                "annotation_status": status,
            })
            queue_rows.append({
                "student_id": student_id,
                "review_type": "reference_vs_pdf_ocr_crosscheck",
                "page_number": top_page,
                "question_id": question_id,
                "subquestion_id": subquestion_id,
                "page_reference_similarity": round(top_score, 2),
                "crop_reference_similarity": round(cross_score, 2),
                "annotation_status": status,
                "reason": "Automated PDF/OCR/reference cross-check; independent visual adjudication still recommended"
                if status != "automatically_cross_verified_pending_independent_review"
                else "Direct label detected and OCR/reference agreement found",
            })

    fields = list(result_rows[0])
    with (OUT / "answer_region_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)

    queue_fields = list(queue_rows[0])
    with (OUT / "mapping_review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=queue_fields)
        writer.writeheader()
        writer.writerows(queue_rows)

    counts = {}
    for row in result_rows:
        counts[row["annotation_status"]] = counts.get(row["annotation_status"], 0) + 1
    print(json.dumps({"subquestion_records": len(result_rows), "status_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
