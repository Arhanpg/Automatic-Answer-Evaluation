"""
backend/cache_manager.py -- Content-Addressed SHA-256 Caching Engine.

Allows previously rendered PDFs, 300 DPI page crops, embeddings, and OCR transcriptions
to load in <15 ms from disk/SQLite.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from backend.database import (
    get_file_cache,
    get_ocr_cache,
    upsert_file_cache,
    upsert_ocr_cache,
)
from config import BASE_DIR, DATA_DIR

logger = logging.getLogger("cache_manager")

CACHE_DIR = os.path.join(DATA_DIR, "cache")
PAGES_CACHE_DIR = os.path.join(CACHE_DIR, "pages")
CROPS_CACHE_DIR = os.path.join(CACHE_DIR, "crops")

os.makedirs(PAGES_CACHE_DIR, exist_ok=True)
os.makedirs(CROPS_CACHE_DIR, exist_ok=True)


def compute_sha256(file_path_or_bytes: Any) -> str:
    """Computes a SHA-256 hex digest of a file path or bytes."""
    hasher = hashlib.sha256()
    if isinstance(file_path_or_bytes, (str, Path)):
        with open(file_path_or_bytes, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    elif isinstance(file_path_or_bytes, (bytes, bytearray)):
        hasher.update(file_path_or_bytes)
    else:
        raise ValueError("Expected file path or bytes for SHA-256 calculation.")
    return hasher.hexdigest()


class CacheManager:
    """Manages high-speed disk and SQLite caching for PDFs and OCR transcriptions."""

    def __init__(self, cache_root: str = CACHE_DIR):
        self.cache_root = cache_root
        self.pages_dir = os.path.join(cache_root, "pages")
        self.crops_dir = os.path.join(cache_root, "crops")
        os.makedirs(self.pages_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)

    def get_or_render_pages(
        self,
        pdf_path: str,
        dpi: int = 300,
        max_width: int = 2048,
        force_recompute: bool = False,
    ) -> Tuple[str, List[str]]:
        """
        Computes SHA-256 of the PDF. Checks if rendered pages exist.
        If cached and not force_recompute, returns cached image paths immediately (<15ms).
        Otherwise renders pages at 300 DPI (max_width 2048) and caches them as PNGs.
        """
        file_sha256 = compute_sha256(pdf_path)
        file_pages_dir = os.path.join(self.pages_dir, file_sha256)
        os.makedirs(file_pages_dir, exist_ok=True)

        cached_file_entry = get_file_cache(file_sha256)

        # Check existing rendered pages
        if not force_recompute and cached_file_entry:
            page_count = cached_file_entry.get("page_count", 0)
            existing_pngs = [
                os.path.join(file_pages_dir, f"page_{i:02d}.png")
                for i in range(page_count)
            ]
            if page_count > 0 and all(os.path.exists(p) for p in existing_pngs):
                logger.debug("Page cache HIT for SHA-256: %s (%d pages)", file_sha256[:8], page_count)
                return file_sha256, existing_pngs

        # Render from PDF
        logger.info("Rendering PDF pages for %s (SHA: %s)...", os.path.basename(pdf_path), file_sha256[:8])
        doc = fitz.open(pdf_path)
        page_paths = []
        page_count = len(doc)

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_idx in range(page_count):
            out_png = os.path.join(file_pages_dir, f"page_{page_idx:02d}.png")
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Limit max_width if necessary while preserving aspect ratio
            if img.width > max_width:
                new_h = int(img.height * (max_width / img.width))
                img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)

            # Enforce RGB conversion before saving
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(out_png, format="PNG", optimize=True)
            page_paths.append(out_png)

        doc.close()

        # Update file cache table
        upsert_file_cache(
            file_sha256=file_sha256,
            file_name=os.path.basename(pdf_path),
            page_count=page_count,
            metadata={"dpi": dpi, "max_width": max_width},
        )

        return file_sha256, page_paths

    def get_cached_ocr(
        self,
        file_sha256: str,
        engine_version: str = "v2_hybrid_consensus",
        force_recompute: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached OCR payload from SQLite ocr_cache table in <15ms.
        """
        if force_recompute:
            return None
        cache_key = f"{file_sha256}_{engine_version}"
        cached = get_ocr_cache(cache_key)
        if cached:
            logger.debug("OCR cache HIT for key: %s", cache_key[:16])
            return cached.get("ocr_payload")
        return None

    def save_cached_ocr(
        self,
        file_sha256: str,
        student_id: str,
        ocr_payload: Dict[str, Any],
        engine_version: str = "v2_hybrid_consensus",
        cer: float = 0.0,
        confidence: float = 1.0,
    ):
        """
        Saves verified OCR payload to SQLite ocr_cache table.
        """
        cache_key = f"{file_sha256}_{engine_version}"
        upsert_ocr_cache(
            cache_key=cache_key,
            file_sha256=file_sha256,
            student_id=student_id,
            engine_version=engine_version,
            ocr_payload=ocr_payload,
            cer=cer,
            confidence=confidence,
        )
        logger.debug("Saved OCR cache for key: %s (student: %s)", cache_key[:16], student_id)


# Global singleton instance
cache_manager = CacheManager()
