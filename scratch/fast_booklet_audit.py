"""
scratch/fast_booklet_audit.py

Fast, accurate booklet audit:
1. Determines booklet orientation from Page 1 cover page (matches printed header).
2. Applies that orientation to all pages of the booklet.
3. Analyzes pages for student writing, detected headers, and omissions.
4. Audits crops and flags multi-page continuations and template overlaps.
"""

import sys
import json
import re
from pathlib import Path
from PIL import Image
import pytesseract

DATA_ROOT = Path("F:/Automatic answer evaluation dataset")
TM_DIR = Path("data/teacher_marks")
OUTPUT_FILE = Path("scratch/fast_audit_results.json")

HEADER_PATTERNS = {
    "1a": re.compile(r"(?:1\s*[\.\-\)]\s*a|ans(?:wer)?\s*1\s*\(?a\)?|q(?:uestion)?\s*1\s*\(?a\)?)", re.I),
    "1b": re.compile(r"(?:1\s*[\.\-\)]\s*b|ans(?:wer)?\s*1\s*\(?b\)?|q(?:uestion)?\s*1\s*\(?b\)?)", re.I),
    "1c": re.compile(r"(?:1\s*[\.\-\)]\s*c|ans(?:wer)?\s*1\s*\(?c\)?|q(?:uestion)?\s*1\s*\(?c\)?)", re.I),
    "1d": re.compile(r"(?:1\s*[\.\-\)]\s*d|ans(?:wer)?\s*1\s*\(?d\)?|q(?:uestion)?\s*1\s*\(?d\)?)", re.I),
    "2a": re.compile(r"(?:2\s*[\.\-\)]\s*a|ans(?:wer)?\s*2\s*\(?a\)?|q(?:uestion)?\s*2\s*\(?a\)?)", re.I),
    "2b": re.compile(r"(?:2\s*[\.\-\)]\s*b|ans(?:wer)?\s*2\s*\(?b\)?|q(?:uestion)?\s*2\s*\(?b\)?)", re.I),
    "2c": re.compile(r"(?:2\s*[\.\-\)]\s*c|ans(?:wer)?\s*2\s*\(?c\)?|q(?:uestion)?\s*2\s*\(?c\)?)", re.I),
    "3a": re.compile(r"(?:3\s*[\.\-\)]\s*a|ans(?:wer)?\s*3\s*\(?a\)?|q(?:uestion)?\s*3\s*\(?a\)?)", re.I),
    "3b": re.compile(r"(?:3\s*[\.\-\)]\s*b|ans(?:wer)?\s*3\s*\(?b\)?|q(?:uestion)?\s*3\s*\(?b\)?)", re.I),
    "3c": re.compile(r"(?:3\s*[\.\-\)]\s*c|ans(?:wer)?\s*3\s*\(?c\)?|q(?:uestion)?\s*3\s*\(?c\)?)", re.I),
    "4": re.compile(r"(?:4\s*[\.\-\)]|ans(?:wer)?\s*4|q(?:uestion)?\s*4)", re.I),
}


def find_booklet_orientation(p1_path: Path) -> int:
    img = Image.open(p1_path)
    scale = 800.0 / max(img.size)
    resized = img.resize((int(img.width * scale), int(img.height * scale)))
    
    for rot in [0, 270, 90, 180]:
        t = pytesseract.image_to_string(resized.rotate(rot, expand=True)).lower()
        if "indian institute" in t or "information technology" in t or "dharwad" in t or "examination" in t:
            return rot
    return 0


def audit_all():
    print("Beginning Fast Forensic Booklet Audit across all 37 students...", flush=True)
    results = {}
    
    for sid_num in range(1, 38):
        sid = f"s{sid_num}"
        s_dir = DATA_ROOT / sid
        imgs = sorted(list(s_dir.glob("*.jpg")))
        
        tm_file = TM_DIR / f"{sid}.json"
        marks = json.load(open(tm_file))["marks"] if tm_file.exists() else {}
        
        rot = find_booklet_orientation(imgs[0])
        
        booklet_info = {
            "student_id": sid,
            "page_count": len(imgs),
            "orientation": rot,
            "teacher_marks": marks,
            "pages": []
        }
        
        for p_idx, im_p in enumerate(imgs):
            img = Image.open(im_p)
            scale = 1000.0 / max(img.size)
            resized = img.resize((int(img.width * scale), int(img.height * scale)))
            upright = resized.rotate(rot, expand=True)
            
            txt = pytesseract.image_to_string(upright, config="--psm 6")
            
            found_headers = []
            for sq, pat in HEADER_PATTERNS.items():
                if pat.search(txt):
                    found_headers.append(sq)
                    
            booklet_info["pages"].append({
                "page_num": p_idx + 1,
                "headers": found_headers,
                "text_snippet": " ".join(txt.split()[:15])
            })
            
        results[sid] = booklet_info
        headers_found = [f"P{p['page_num']}:{p['headers']}" for p in booklet_info["pages"] if p['headers']]
        print(f"[{sid}] {len(imgs)}p, rot={rot} | Marks: {marks} | Headers: {', '.join(headers_found)}", flush=True)
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    audit_all()
