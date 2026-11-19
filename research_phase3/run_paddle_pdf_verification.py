"""Run real PaddleOCR across every rendered answer-page image."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import ocr_engine

OUT = ROOT / "research_phase3"
IMAGE_DIR = OUT / "answer_region_images"


def main() -> None:
    results = {}
    images = sorted(IMAGE_DIR.glob("*_page_[0-9][0-9][0-9].png"))
    for index, image_path in enumerate(images, 1):
        match = re.match(r"(.+)_page_(\d{3})\.png$", image_path.name)
        if not match:
            continue
        student_id, page_number = match.group(1), int(match.group(2))
        try:
            with Image.open(image_path) as pil_image:
                lines = ocr_engine.merge_rows(
                    ocr_engine.ocr_image(pil_image),
                    page=page_number - 1,
                )
            text = " ".join(row["text"] for row in lines)
            status = "executed"
            mean_confidence = (
                sum(float(row["conf"]) for row in lines) / len(lines)
                if lines else 0.0
            )
        except Exception as exc:
            lines = []
            text = ""
            status = f"{type(exc).__name__}: {exc}"
            mean_confidence = 0.0

        results[f"{student_id}|{page_number}"] = {
            "student_id": student_id,
            "page_number": page_number,
            "execution_status": status,
            "mean_confidence": mean_confidence,
            "text": text,
            "lines": lines,
        }
        if index % 10 == 0 or index == len(images):
            print(json.dumps({"completed": index, "total": len(images), "last": image_path.name}))

    output = OUT / "paddle_page_ocr.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
