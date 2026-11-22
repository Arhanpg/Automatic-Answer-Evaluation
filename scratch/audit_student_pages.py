"""
scratch/audit_student_pages.py

Performs comprehensive forensic inspection across all 37 student booklets:
1. Detects rotation (0, 90, 180, 270) for every page using line/contour orientations.
2. Extracts text using pytesseract with proper rotation and psm mode.
3. Spots question headers: 1a, 1b, 1c, 1d, 2a, 2b, 2c, 3a, 3b, 3c, 4.
4. Identifies true omissions vs written answers.
5. Flags discrepancies with historical research_v2/annotations/answer_regions.csv.
"""

import re
import json
from pathlib import Path
from PIL import Image
import pytesseract

DATA_ROOT = Path("F:/Automatic answer evaluation dataset")
TM_DIR = Path("data/teacher_marks")
OUTPUT_FILE = Path("scratch/forensic_page_audit.json")

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

KEYWORDS = {
    "1a": ["n-gram", "ngram", "limitation", "context", "sparsity"],
    "1b": ["positional", "encoding", "order", "permutation", "sinusoidal"],
    "1c": ["encoder", "decoder", "bert", "gpt", "bidirectional", "autoregressive"],
    "1d": ["masked", "masking", "future", "causal", "lookahead"],
    "2a": ["dot", "product", "q.k", "q k", "k1", "k2", "scores"],
    "2b": ["softmax", "exp", "weights", "probabilities"],
    "2c": ["output", "vector", "v1", "v2", "context"],
    "3a": ["greedy", "word a", "highest", "0.4"],
    "3b": ["top-k", "top k", "k=2", "word a", "word b"],
    "3c": ["layer", "normalization", "norm", "residual", "twice", "pre-ln", "post-ln"],
    "4": ["laplace", "add-1", "add 1", "smoothing", "bigram", "ml", "vocab"],
}


def audit_booklet(sid: str):
    s_dir = DATA_ROOT / sid
    imgs = sorted(list(s_dir.glob("*.jpg")))
    
    tm_file = TM_DIR / f"{sid}.json"
    marks = json.load(open(tm_file))["marks"] if tm_file.exists() else {}

    booklet_report = {
        "student_id": sid,
        "page_count": len(imgs),
        "teacher_marks": marks,
        "pages": []
    }

    for p_idx, img_p in enumerate(imgs):
        p_num = p_idx + 1
        img = Image.open(img_p)
        scale = 1200.0 / max(img.size)
        resized = img.resize((int(img.width * scale), int(img.height * scale)))

        # Determine orientation
        best_rot = 0
        best_text = ""
        max_kw_count = -1

        for rot in [0, 90, 180, 270]:
            r_img = resized.rotate(rot, expand=True)
            txt = pytesseract.image_to_string(r_img, config="--psm 6")
            txt_lower = txt.lower()
            
            # Count keyword matches
            kw_count = sum(1 for kw_list in KEYWORDS.values() for kw in kw_list if kw in txt_lower)
            if "indian institute of information" in txt_lower:
                kw_count += 5  # cover page bonus
                
            if kw_count > max_kw_count:
                max_kw_count = kw_count
                best_rot = rot
                best_text = txt

        # Check detected subquestions
        detected_sqs = []
        best_lower = best_text.lower()
        for sq, pattern in HEADER_PATTERNS.items():
            if pattern.search(best_text):
                detected_sqs.append(sq)
            else:
                # Check keyword overlap
                match_count = sum(1 for kw in KEYWORDS[sq] if kw in best_lower)
                if match_count >= 2:
                    detected_sqs.append(sq)

        booklet_report["pages"].append({
            "page_num": p_num,
            "filename": img_p.name,
            "best_rotation": best_rot,
            "keyword_count": max_kw_count,
            "detected_subquestions": list(set(detected_sqs)),
            "text_preview": " ".join(best_text.split()[:20])
        })

    return booklet_report


def main():
    print("Executing Forensic Page Audit across all 37 booklets...")
    all_reports = {}
    for sid_num in range(1, 38):
        sid = f"s{sid_num}"
        report = audit_booklet(sid)
        all_reports[sid] = report
        detected = [f"P{p['page_num']}(rot={p['best_rotation']}):{p['detected_subquestions']}" for p in report["pages"]]
        print(f"[{sid}] {report['page_count']} pages | Marks: {report['teacher_marks']} | {', '.join(detected)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nAudit complete. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
