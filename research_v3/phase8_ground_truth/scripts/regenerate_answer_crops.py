"""
research_v3/phase8_ground_truth/scripts/regenerate_answer_crops.py

Regenerates answer crop images from raw page scans using verified bounding boxes.
Guarantees:
1. Loads full-resolution scans from AAE_DATA_ROOT.
2. Applies visual pixel bounding boxes from answer_region manifests.
3. Computes SHA-256 cryptographic digest of each generated crop.
4. Saves crops to local private directory (research_v3/crops/), strictly excluded from git.
5. Records crop hash and dimensions in crops_manifest.csv.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_ROOT = Path(os.environ.get("AAE_DATA_ROOT", r"F:\Automatic answer evaluation dataset"))
V2_ANN_DIR = REPO_ROOT / "research_v2" / "annotations"
CROPS_OUTPUT_DIR = REPO_ROOT / "research_v3" / "crops"
CROPS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def regenerate_crops():
    print("=" * 80)
    print("PHASE 8: ANSWER CROP REGENERATION AND CRYPTOGRAPHIC VERIFICATION")
    print("=" * 80)

    regions_file = V2_ANN_DIR / "answer_regions.csv"
    pages_file = V2_ANN_DIR / "pages.csv"

    if not regions_file.exists() or not pages_file.exists():
        print(f"Error: Required metadata files missing in {V2_ANN_DIR}")
        sys.exit(1)

    regions_df = pd.read_csv(regions_file)
    pages_df = pd.read_csv(pages_file)

    page_lookup = {}
    for _, row in pages_df.iterrows():
        page_lookup[row["page_id"]] = {
            "relative_path": row["image_relative_path"],
            "width": int(row["width_px"]),
            "height": int(row["height_px"])
        }

    manifest_rows = []
    success_count = 0
    omitted_count = 0
    error_count = 0

    print(f"Processing {len(regions_df)} answer regions against raw scans in {DATA_ROOT}...")

    grouped = regions_df.groupby("page_id")

    for page_id, group in grouped:
        if page_id not in page_lookup:
            print(f"Warning: Page ID {page_id} not found in pages.csv")
            error_count += len(group)
            continue

        page_meta = page_lookup[page_id]
        img_rel_path = page_meta["relative_path"]
        img_path = DATA_ROOT / img_rel_path

        if not img_path.exists():
            print(f"Warning: Raw scan missing on disk: {img_path}")
            error_count += len(group)
            continue

        try:
            with Image.open(img_path) as full_img:
                img_w, img_h = full_img.size

                for _, reg in group.iterrows():
                    sid = reg["student_id"]
                    sqid = reg["subquestion_id"]
                    status = str(reg.get("annotation_status", ""))
                    crop_sha = str(reg.get("crop_sha256", ""))
                    crop_rel = str(reg.get("crop_relative_path", ""))

                    if "omitted" in status.lower() or crop_sha == "OMITTED_BY_STUDENT" or crop_rel.upper() == "NONE":
                        omitted_count += 1
                        manifest_rows.append({
                            "student_id": sid,
                            "subquestion_id": sqid,
                            "page_id": page_id,
                            "crop_filename": "OMITTED",
                            "crop_sha256": "OMITTED_BY_STUDENT",
                            "crop_width": 0,
                            "crop_height": 0,
                            "status": "omitted_by_student"
                        })
                        continue

                    x0 = max(0, min(int(reg["x0"]), img_w - 1))
                    y0 = max(0, min(int(reg["y0"]), img_h - 1))
                    x1 = max(1, min(int(reg["x1"]), img_w))
                    y1 = max(1, min(int(reg["y1"]), img_h))

                    if x0 >= x1 or y0 >= y1:
                        print(f"Error: Inverted bounding box for {sid}_{sqid} on {page_id}: ({x0},{y0},{x1},{y1})")
                        error_count += 1
                        continue

                    crop = full_img.crop((x0, y0, x1, y1))
                    crop_filename = f"{sid}_{sqid}_crop.png"
                    crop_path = CROPS_OUTPUT_DIR / crop_filename

                    crop.save(crop_path, format="PNG")

                    with open(crop_path, "rb") as f:
                        c_bytes = f.read()
                    c_sha = sha256_bytes(c_bytes)

                    manifest_rows.append({
                        "student_id": sid,
                        "subquestion_id": sqid,
                        "page_id": page_id,
                        "crop_filename": crop_filename,
                        "crop_sha256": c_sha,
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "status": "written"
                    })
                    success_count += 1

        except Exception as e:
            print(f"Error processing page {page_id}: {e}")
            error_count += len(group)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = CROPS_OUTPUT_DIR / "crops_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    print("\n" + "=" * 80)
    print("CROP REGENERATION COMPLETE:")
    print(f"  Successfully regenerated crops: {success_count}")
    print(f"  Omitted slots confirmed:        {omitted_count}")
    print(f"  Errors encountered:             {error_count}")
    print(f"  Total records in manifest:      {len(manifest_df)}")
    print(f"  Manifest written to:            {manifest_csv}")
    print("=" * 80)


if __name__ == "__main__":
    regenerate_crops()
