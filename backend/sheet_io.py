"""
backend/sheet_io.py -- find student scripts in a folder and load their pages.

Supported layouts (all can be mixed in one folder):
  A. folder/s1.pdf, s2.pdf ...            one multi-page PDF per student   <- your layout
  B. folder/s1/*.jpg, s2/*.jpg ...        one sub-folder of page photos per student
  C. folder/21CS045_1.jpg, 21CS045_2.jpg  pages named <id>_<page>  (grouped by id)
  D. folder/s1.jpg, s2.jpg ...            one image per student
The student id is the file / folder name (config.ROLL_SOURCE = "filename").
"""
import os
import re

from backend import pdf_utils
import config
from backend import image_utils

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jfif"}
PDF_EXT = {".pdf"}
_PAGE_RE = re.compile(r"^(?P<id>.+?)[_\-\s](?:p|pg|page)?(?P<n>\d{1,3})$", re.I)


def natural_key(s):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def _ext(name):
    return os.path.splitext(name)[1].lower()


def collect_units(folder):
    """-> [{"roll": "s1", "files": [paths in page order]}] sorted naturally."""
    units = {}

    def add(roll, path):
        units.setdefault(roll, []).append(path)

    entries = sorted(os.listdir(folder), key=natural_key)
    for e in entries:
        full = os.path.join(folder, e)
        if e.startswith(".") or e.startswith("__"):
            continue
        if os.path.isdir(full):
            for f in sorted(os.listdir(full), key=natural_key):
                if _ext(f) in IMAGE_EXT | PDF_EXT and os.path.isfile(os.path.join(full, f)):
                    add(e, os.path.join(full, f))
        elif _ext(e) in PDF_EXT:
            add(os.path.splitext(e)[0], full)
        elif _ext(e) in IMAGE_EXT:
            stem = os.path.splitext(e)[0]
            m = _PAGE_RE.match(stem)
            add(m.group("id") if m else stem, full)
    return [{"roll": r, "files": sorted(fs, key=lambda p: natural_key(os.path.basename(p)))}
            for r, fs in sorted(units.items(), key=lambda kv: natural_key(kv[0]))]


def load_pages(unit):
    pages = []
    for f in unit["files"]:
        if _ext(f) in PDF_EXT:
            pages.extend(pdf_utils.render_pdf(f, config.OCR_DPI))
        else:
            pages.append(pdf_utils.load_image(f))
    return pages


def pages_dir(roll):
    return os.path.join(config.PAGES_DIR, roll)


def save_page_images(roll, pages):
    d = pages_dir(roll)
    os.makedirs(d, exist_ok=True)
    for old in os.listdir(d):
        os.remove(os.path.join(d, old))
    for i, img in enumerate(pages, start=1):
        image_utils.save_page_jpeg(img, os.path.join(d, f"p{i}.jpg"))
    return len(pages)


def count_saved_pages(roll):
    d = pages_dir(roll)
    return len([f for f in os.listdir(d) if f.endswith(".jpg")]) if os.path.isdir(d) else 0
