import os
import sys
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from backend import ocr_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER = r"F:\Automatic answer evaluation dataset\All PDF's"
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "ground_truth", "real_booklet_covers.json")

all_pdf_files = sorted(
    [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")],
    key=lambda x: int("".join(c for c in x if c.isdigit()) or 0)
)

print(f"Loaded {len(all_pdf_files)} PDF scripts from {PDF_FOLDER}")

# Build engine once
engine = ocr_engine._build_engine()

results = {}

for idx, pdf in enumerate(all_pdf_files, 1):
    sid = os.path.splitext(pdf)[0]
    # Check cover page: usually p1, except s2 which has cover on p2
    cover_page = "p2.jpg" if sid == "s2" else "p1.jpg"
    img_path = os.path.join(BASE_DIR, "data", "pages", sid, cover_page)
    
    if not os.path.exists(img_path):
        print(f"[{idx}/37] {sid}: Image not found at {img_path}")
        continue
        
    img = Image.open(img_path)
    lines = ocr_engine.ocr_image(img)
    
    # Parse lines
    student_name = ""
    reg_no = ""
    serial_no = ""
    total_marks = None
    q_marks = {"Q1": None, "Q2": None, "Q3": None, "Q4": None}
    
    # First pass: student info
    for ln in lines:
        t = ln["text"].strip()
        bbox = ln["bbox"]
        
        # Serial number (top right)
        if bbox[1] < 450 and bbox[0] > 700:
            m_sl = re.search(r"\b(\d{5,8})\b", t)
            if m_sl and not serial_no:
                serial_no = m_sl.group(1)
                
        # Registration number
        m_reg = re.search(r"\b(23B[A-Za-z0-9]{2,4}\d{2,4})\b", t, re.I)
        if m_reg and not reg_no:
            reg_no = m_reg.group(1).upper()
            
        # Name: line after 'Name of the Student' or matches capitalized names
        if 350 < bbox[1] < 460 and 250 < bbox[0] < 700:
            if not any(k in t.lower() for k in ["name", "student", "signature", "sem"]):
                if len(t) > 3 and not student_name:
                    student_name = t
                    
        # Total in figures (bottom of examiner table)
        if "total in figures" in t.lower() or ("total" in t.lower() and bbox[1] > 1250):
            pass
        if bbox[1] > 1300 and bbox[0] > 750:
            m_tot = re.search(r"\b(\d{1,2}(?:\.\d)?)\b", t)
            if m_tot and total_marks is None:
                val = float(m_tot.group(1))
                if val <= 20.0 and val > 0:
                    total_marks = val

    results[sid] = {
        "student_id": sid,
        "student_name": student_name,
        "registration_no": reg_no,
        "serial_no": serial_no,
        "total_marks": total_marks,
        "lines": lines
    }
    print(f"[{idx}/37] {sid}: Name='{student_name}' | Reg='{reg_no}' | Sl='{serial_no}' | Total={total_marks}")

# Save raw output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    # Save without raw lines for clean view
    clean_summary = {k: {sk: v[sk] for sk in v if sk != "lines"} for k, v in results.items()}
    json.dump(clean_summary, f, indent=2)

print(f"Saved real booklet covers summary to {OUTPUT_FILE}")
