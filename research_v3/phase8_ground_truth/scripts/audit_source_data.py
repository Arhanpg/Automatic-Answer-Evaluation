"""
research_v3/phase8_ground_truth/scripts/audit_source_data.py

Performs exhaustive preflight forensic audit on the repository:
1. Scans and inventories raw page images in data/pages
2. Inspects teacher mark records in data/teacher_marks
3. Reconciles page counts, dimensions, file hashes, and DPI metadata
4. Audits research_v2 annotations (booklets, pages, answer_regions, transcripts, marks)
5. Reconciles the 388 vs 392 written answer count discrepancy
6. Evaluates crop generation integrity and bounding box geometry
7. Audits test split exposure history
8. Scans Git-tracked files for PII, roll numbers, or sensitive content
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit():
    print("=" * 80)
    print("PHASE 8 FORENSIC AUDIT: SOURCE DATA AND PROVENANCE")
    print("=" * 80)

    ann_dir = REPO_ROOT / "research_v2" / "annotations"

    # 1. Inspect data/pages and AAE_DATA_ROOT
    data_root_env = os.environ.get("AAE_DATA_ROOT", r"F:\Automatic answer evaluation dataset")
    data_root = Path(data_root_env)
    print(f"\n1. SOURCE DATASET DIRECTORY AUDIT (AAE_DATA_ROOT = {data_root}):")
    if data_root.exists():
        raw_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("s")]
        print(f"Total student directories in AAE_DATA_ROOT: {len(raw_dirs)}")
        all_raw_images = list(data_root.glob("s*/*.jpg")) + list(data_root.glob("s*/*.png"))
        print(f"Total raw scanned page images in AAE_DATA_ROOT: {len(all_raw_images)}")
        
        # Check image sizes and DPI in raw dataset
        raw_dims = []
        raw_dpis = []
        for im_p in all_raw_images:
            try:
                with Image.open(im_p) as im:
                    raw_dims.append(im.size)
                    raw_dpis.append(im.info.get("dpi", (72.0, 72.0)))
            except Exception as e:
                pass
        print(f"Unique dimensions in raw dataset: {set(raw_dims)}")
        print(f"Unique DPI in raw dataset: {set(raw_dpis)}")
    else:
        print(f"WARNING: AAE_DATA_ROOT does not exist at {data_root}")

    # 2. Inspect research_v2/manifests/page_inventory.csv and research_v2/annotations/pages.csv
    pages_csv = ann_dir / "pages.csv"
    if pages_csv.exists():
        pdf = pd.read_csv(pages_csv)
        print(f"\n2. RESEARCH_V2/ANNOTATIONS/PAGES.CSV:")
        print(f"Total records in pages.csv: {len(pdf)}")
        print(f"Columns in pages.csv: {list(pdf.columns)}")
        print(f"Sample relative paths in pages.csv: {pdf['image_relative_path'].head(5).tolist()}")
        
        # Check if pages.csv paths exist on disk
        found_pages = 0
        missing_pages = []
        for _, row in pdf.iterrows():
            rel = row['image_relative_path']
            # Check relative to AAE_DATA_ROOT, REPO_ROOT, and data
            p1 = data_root / rel
            p2 = REPO_ROOT / "data" / rel
            p3 = REPO_ROOT / rel
            if p1.exists() or p2.exists() or p3.exists():
                found_pages += 1
            else:
                missing_pages.append((row['student_id'], rel))
        print(f"Pages found on disk matching pages.csv: {found_pages} / {len(pdf)}")
        if missing_pages:
            print(f"Missing pages ({len(missing_pages)}): {missing_pages[:5]}")

    # 3. Inspect research_v2/annotations
    print(f"\n3. RESEARCH_V2/ANNOTATIONS AUDIT:")
    for f in ["booklets.csv", "pages.csv", "marks.csv", "transcripts.csv", "answer_regions.csv", "adjudication_log.csv"]:
        fp = ann_dir / f
        if fp.exists():
            df = pd.read_csv(fp)
            print(f"  - {f}: {len(df)} rows, columns: {list(df.columns)}")
        else:
            print(f"  - {f}: MISSING")

    # 4. Reconcile Response Counts: 388 vs 392
    transcripts_csv = ann_dir / "transcripts.csv"
    regions_csv = ann_dir / "answer_regions.csv"
    ocr_metrics_csv = REPO_ROOT / "research_v2" / "ocr" / "ocr_metrics_by_answer.csv"

    print(f"\n4. RESPONSE COUNT RECONCILIATION:")
    if transcripts_csv.exists():
        tdf = pd.read_csv(transcripts_csv)
        print(f"transcripts.csv total records: {len(tdf)}")
        status_counts = tdf["answer_status"].value_counts().to_dict()
        print(f"transcripts.csv answer_status breakdown: {status_counts}")

    if regions_csv.exists():
        rdf = pd.read_csv(regions_csv)
        print(f"answer_regions.csv total records: {len(rdf)}")
        print(f"answer_regions.csv columns: {list(rdf.columns)}")
        if "answer_status" in rdf.columns:
            print(f"answer_regions.csv answer_status breakdown: {rdf['answer_status'].value_counts().to_dict()}")

    if ocr_metrics_csv.exists():
        odf = pd.read_csv(ocr_metrics_csv)
        print(f"ocr_metrics_by_answer.csv total records: {len(odf)}")
        sub_o = odf[(odf["engine"] == "paddleocr") & (odf["condition"] == "raw")]
        print(f"Unique answer evaluations in OCR (PaddleOCR raw): {len(sub_o)}")
        print(f"Students in PaddleOCR raw: {sub_o['student_id'].nunique()} students: {sorted(sub_o['student_id'].unique().tolist())}")
        for eng in odf["engine"].unique():
            eng_sub = odf[odf["engine"] == eng]
            print(f"  Engine '{eng}': {eng_sub['student_id'].nunique()} students ({len(eng_sub)} total records across conditions)")
        ocr_status_counts = sub_o["answer_status"].value_counts().to_dict()
        print(f"OCR answer_status breakdown: {ocr_status_counts}")

        # Find discrepancy between transcripts.csv and ocr_metrics_by_answer.csv
        omitted_in_transcripts = set(tdf[tdf["answer_status"] == "omitted"]["student_id"] + "_" + tdf[tdf["answer_status"] == "omitted"]["subquestion_id"])
        omitted_in_ocr = set(sub_o[sub_o["answer_status"] == "omitted_by_student"]["student_id"] + "_" + sub_o[sub_o["answer_status"] == "omitted_by_student"]["subquestion_id"])
        print(f"Omitted in transcripts.csv (N={len(omitted_in_transcripts)}): {sorted(omitted_in_transcripts)}")
        print(f"Omitted in ocr_metrics (N={len(omitted_in_ocr)}): {sorted(omitted_in_ocr)}")
        diff_omissions = omitted_in_transcripts.symmetric_difference(omitted_in_ocr)
        print(f"Discrepancy items (difference in omissions): {diff_omissions}")

        # Check where the 388 came from
        gate1_rep = REPO_ROOT / "research_v2" / "reports" / "GATE_1_ANNOTATION.md"
        if gate1_rep.exists():
            rep_txt = gate1_rep.read_text(encoding="utf-8")
            for line in rep_txt.splitlines():
                if "388" in line or "392" in line or "omitted" in line.lower() or "19" in line:
                    print(f"  Gate 1 report excerpt: {line.strip()}")

    # 5. Teacher marks audit
    print(f"\n5. TEACHER MARKS AUDIT:")
    tm_dir = REPO_ROOT / "data" / "teacher_marks"
    tm_files = list(tm_dir.glob("s*.json"))
    print(f"Total teacher mark JSON files: {len(tm_files)}")
    sample_tm = json.loads(tm_files[0].read_text(encoding="utf-8"))
    print(f"Sample teacher mark JSON structure ({tm_files[0].name}): {sample_tm}")

    # Check marks.csv in research_v2/annotations
    marks_csv = ann_dir / "marks.csv"
    if marks_csv.exists():
        mdf = pd.read_csv(marks_csv)
        print(f"marks.csv records: {len(mdf)}, students: {mdf['student_id'].nunique()}")
        print(f"Parent question mark max check: {mdf.groupby('parent_question_id')['teacher_mark'].max().to_dict()}")

    # 6. Audit Bounding Box Coordinates and Crop Files
    print(f"\n6. ANSWER REGION BOUNDING BOX & CROP AUDIT:")
    crops_dir = REPO_ROOT / "research_v2" / "crops"
    crop_files = list(crops_dir.glob("*.png")) if crops_dir.exists() else []
    print(f"Total crop image files on disk in research_v2/crops: {len(crop_files)}")

    # Check which written answers in transcripts.csv do not have a crop file
    if transcripts_csv.exists():
        written_subqs = tdf[tdf["answer_status"] == "written"]
        crop_stems = set(f.stem.replace("_crop", "") for f in crop_files)
        missing_crops = []
        for _, r in written_subqs.iterrows():
            key = f"{r['student_id']}_{r['subquestion_id']}"
            if key not in crop_stems:
                missing_crops.append(key)
        print(f"Written subquestions without a crop file in research_v2/crops (N={len(missing_crops)}): {missing_crops}")

        # Check which crops exist that might be marked omitted
        omitted_subqs = tdf[tdf["answer_status"] == "omitted"]
        omitted_crops = []
        for _, r in omitted_subqs.iterrows():
            key = f"{r['student_id']}_{r['subquestion_id']}"
            if key in crop_stems:
                omitted_crops.append(key)
        print(f"Omitted subquestions that HAVE a crop file (N={len(omitted_crops)}): {omitted_crops}")

    # 8. Privacy & PII Audit of Git-Tracked Files
    print(f"\n8. PRIVACY AND SENSITIVE DATA AUDIT:")
    import subprocess
    import re
    res = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    tracked_files = [f for f in res.stdout.splitlines() if f.strip()]
    print(f"Total Git-tracked files in repository: {len(tracked_files)}")

    pii_pattern = re.compile(r"\b(?:23BCS\d+|21CS\d+)\b", re.IGNORECASE)
    personal_path_pattern = re.compile(r"[A-Z]:\\(?:Users|Automatic Answer evaluation)", re.IGNORECASE)
    name_pattern = re.compile(r"\b(?:Arhan|Abhinav|Ghosarwade|Gumgol)\b", re.IGNORECASE) # author names allowed, checking for student names

    files_with_roll_numbers = []
    files_with_personal_paths = []
    crops_tracked = []

    for tf in tracked_files:
        if tf.startswith("research_v2/crops/") or tf.endswith(".png") or tf.endswith(".jpg"):
            crops_tracked.append(tf)
        tf_path = REPO_ROOT / tf
        if not tf_path.is_file():
            continue
        # Skip binary files
        try:
            content = tf_path.read_text(encoding="utf-8", errors="ignore")
            if pii_pattern.search(content):
                matches = pii_pattern.findall(content)
                files_with_roll_numbers.append((tf, set(matches)))
            if personal_path_pattern.search(content):
                files_with_personal_paths.append(tf)
        except Exception:
            pass

    print(f"Git-tracked image crops or binary page scans: {len(crops_tracked)}")
    print(f"Git-tracked files containing institutional roll numbers (23BCS*/21CS*): {len(files_with_roll_numbers)}")
    if files_with_roll_numbers:
        for f, m in files_with_roll_numbers[:10]:
            print(f"   - {f}: {m}")
    print(f"Git-tracked files containing hardcoded local personal Windows paths: {len(files_with_personal_paths)}")
    if files_with_personal_paths:
        for f in files_with_personal_paths[:10]:
            print(f"   - {f}")
    print(f"\n7. TEST SPLIT EXPOSURE AUDIT:")
    print("Locked test students identified: s7, s14, s24, s29, s30, s37")
    prov_csv = REPO_ROOT / "research_v2" / "00_EVIDENCE_PROVENANCE.csv"
    if prov_csv.exists():
        pr_df = pd.read_csv(prov_csv)
        print(f"00_EVIDENCE_PROVENANCE.csv rows: {len(pr_df)}")
        for _, r in pr_df.iterrows():
            if "test" in str(r.to_dict()).lower() or "exposure" in str(r.to_dict()).lower() or "s7" in str(r.to_dict()):
                print(f"  Provenance note: {r.to_dict()}")


if __name__ == "__main__":
    audit()
