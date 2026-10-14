"""
backend/pdf_utils.py -- read text layers from, and render pages of, PDFs.

Primary engine is PyMuPDF (you already have it: `import pymupdf`).
Fallbacks keep things working elsewhere: pdfplumber/pypdf for the text
layer, and the `pdftoppm` command-line tool for rendering.
"""
import glob
import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageOps


def _mupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz  # older PyMuPDF name
            return fitz
        except ImportError:
            return None


def pdf_text_pages(path):
    """Text layer, one string per page ('' for a scanned/image-only page)."""
    m = _mupdf()
    if m is not None:
        doc = m.open(path)
        try:
            return [p.get_text("text") for p in doc]
        finally:
            doc.close()
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return [(p.extract_text() or "") for p in pdf.pages]
    except ImportError:
        pass
    try:
        import pypdf
        return [(p.extract_text() or "") for p in pypdf.PdfReader(path).pages]
    except ImportError:
        return []


def has_text_layer(path, min_chars=150):
    return sum(len(t.strip()) for t in pdf_text_pages(path)) >= min_chars


def render_pdf(path, dpi=200):
    """Return every page as an RGB PIL image."""
    m = _mupdf()
    if m is not None:
        zoom = dpi / 72.0
        doc = m.open(path)
        try:
            pages = []
            for page in doc:
                pix = page.get_pixmap(matrix=m.Matrix(zoom, zoom), alpha=False)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            return pages
        finally:
            doc.close()
    if shutil.which("pdftoppm"):
        tmp = tempfile.mkdtemp(prefix="aae_pdf_")
        try:
            subprocess.run(["pdftoppm", "-r", str(dpi), "-png", path, os.path.join(tmp, "p")],
                           check=True, capture_output=True)
            files = sorted(glob.glob(os.path.join(tmp, "p*.png")))
            return [Image.open(f).convert("RGB").copy() for f in files]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise RuntimeError("Cannot render PDFs: install PyMuPDF  ->  pip install pymupdf")


def load_image(path):
    """Open an image file with EXIF rotation applied, as RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB").copy()
