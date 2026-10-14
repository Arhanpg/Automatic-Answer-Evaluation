"""
backend/ocr_engine.py -- PaddleOCR wrapper (works with BOTH PaddleOCR 2.x and 3.x).

What comes out is deliberately low-level: text *rows* with their position on
the page. Turning rows into "answer to Q1(a)" happens in sheet_parser.py,
guided by the question paper -- that is what makes the result usable.

There is NO silent fake-data fallback any more. If PaddleOCR is missing you
get an OCRUnavailable error with install instructions (use AAE_MODE=demo if
you only want to click around the UI).
"""
import os
import tempfile
import threading

import numpy as np

import config
from backend import image_utils

INSTALL_HELP = (
    "PaddleOCR is not installed (or failed to start).\n"
    "  1) Install PaddlePaddle for your CUDA version: "
    "https://www.paddlepaddle.org.cn/en/install/quick\n"
    "  2) pip install paddleocr\n"
    "Then restart the app. (For a UI-only tour without OCR: AAE_MODE=demo)"
)


class OCRUnavailable(RuntimeError):
    pass


_engine = None
_engine_info = ""
_lock = threading.Lock()
_AVAILABLE = None
_TESSERACT_AVAILABLE = None


def paddleocr_available():
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import paddleocr  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def tesseract_available():
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        try:
            import pytesseract
            tess_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tess_cmd):
                pytesseract.pytesseract.tesseract_cmd = tess_cmd
            pytesseract.get_tesseract_version()
            _TESSERACT_AVAILABLE = True
        except Exception:
            _TESSERACT_AVAILABLE = False
    return _TESSERACT_AVAILABLE


def ocr_available():
    return paddleocr_available() or tesseract_available()


def engine_description():
    if paddleocr_available():
        return _engine_info or "PaddleOCR installed"
    if tesseract_available():
        return "Tesseract OCR"
    return "OCR missing"


def _build_engine():
    global _engine, _engine_info
    if _engine is not None:
        return _engine
    if not paddleocr_available():
        if tesseract_available():
            return None
        raise OCRUnavailable(INSTALL_HELP)
    from paddleocr import PaddleOCR

    lang = config.PADDLEOCR_LANG
    want_gpu = config.PADDLEOCR_DEVICE.lower() != "cpu"
    attempts = []
    if want_gpu:
        attempts.append(("PaddleOCR 3.x on GPU", dict(
            lang=lang, device="gpu:0", use_doc_orientation_classify=False,
            use_doc_unwarping=False, use_textline_orientation=False)))
    attempts.append(("PaddleOCR 3.x on CPU", dict(
        lang=lang, device="cpu", use_doc_orientation_classify=False,
        use_doc_unwarping=False, use_textline_orientation=False)))
    attempts.append(("PaddleOCR 2.x", dict(
        lang=lang, use_angle_cls=True, use_gpu=want_gpu, show_log=False)))
    attempts.append(("PaddleOCR (defaults)", dict(lang=lang)))

    errors = []
    for label, kwargs in attempts:
        try:
            _engine = PaddleOCR(**kwargs)
            _engine_info = label
            return _engine
        except Exception as e:  # different PaddleOCR versions reject different kwargs
            errors.append(f"{label}: {type(e).__name__}: {e}")
    raise OCRUnavailable("PaddleOCR could not start:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# result parsing (kept separate so it can be unit-tested without PaddleOCR)
# ---------------------------------------------------------------------------
def _poly_to_bbox(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max())]


def lines_from_result(out):
    """Normalise PaddleOCR 3.x (predict) or 2.x (ocr) output to
    [{"text", "bbox":[x0,y0,x1,y1], "conf"}]."""
    lines = []
    for res in out or []:
        if res is None:
            continue
        # ---- 3.x: result objects / dicts with rec_texts, rec_scores, rec_polys
        data = None
        if isinstance(res, dict):
            data = res.get("res", res)
        elif hasattr(res, "json"):
            j = res.json
            data = j.get("res", j) if isinstance(j, dict) else None
        if isinstance(data, dict) and "rec_texts" in data:
            texts = list(data.get("rec_texts") or [])
            scores = list(data.get("rec_scores") or [1.0] * len(texts))
            polys = data.get("rec_polys")
            if polys is None:
                polys = data.get("dt_polys")
            if polys is None:
                polys = data.get("rec_boxes")
            polys = list(polys) if polys is not None else []
            for t, s, p in zip(texts, scores, polys):
                lines.append({"text": str(t), "bbox": _poly_to_bbox(p), "conf": float(s)})
            continue
        # ---- 2.x: [[poly, (text, conf)], ...] for one image
        for item in res:
            try:
                poly, (text, conf) = item
                lines.append({"text": str(text), "bbox": _poly_to_bbox(poly), "conf": float(conf)})
            except (TypeError, ValueError):
                continue
    return lines


def _tesseract_lines(img):
    import pytesseract
    tess_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n_boxes = len(data.get("text", []))
    line_dict = {}
    for i in range(n_boxes):
        txt = data["text"][i].strip()
        if not txt:
            continue
        conf = float(data["conf"][i]) if float(data["conf"][i]) >= 0 else 50.0
        conf = conf / 100.0
        b_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x0, y0, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        x1, y1 = x0 + w, y0 + h
        if b_key not in line_dict:
            line_dict[b_key] = {"words": [txt], "x0": x0, "y0": y0, "x1": x1, "y1": y1, "confs": [conf]}
        else:
            line_dict[b_key]["words"].append(txt)
            line_dict[b_key]["x0"] = min(line_dict[b_key]["x0"], x0)
            line_dict[b_key]["y0"] = min(line_dict[b_key]["y0"], y0)
            line_dict[b_key]["x1"] = max(line_dict[b_key]["x1"], x1)
            line_dict[b_key]["y1"] = max(line_dict[b_key]["y1"], y1)
            line_dict[b_key]["confs"].append(conf)

    lines = []
    for b_key, val in line_dict.items():
        lines.append({
            "text": " ".join(val["words"]),
            "bbox": [float(val["x0"]), float(val["y0"]), float(val["x1"]), float(val["y1"])],
            "conf": float(sum(val["confs"]) / len(val["confs"])),
        })
    return lines


def ocr_image(img):
    """OCR one PIL image -> list of lines (see lines_from_result)."""
    if paddleocr_available():
        with _lock:
            engine = _build_engine()
            arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()   # RGB -> BGR for Paddle
            if hasattr(engine, "predict"):
                out = engine.predict(arr)
            else:
                out = engine.ocr(arr, cls=True)
        return lines_from_result(out)
    elif tesseract_available():
        return _tesseract_lines(img)
    else:
        raise OCRUnavailable(INSTALL_HELP)


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------
def merge_rows(lines, page=0):
    """Join text boxes that sit on the same visual line, left to right."""
    items = []
    for ln in lines:
        if not ln["text"].strip():
            continue
        x0, y0, x1, y1 = ln["bbox"]
        items.append({**ln, "_yc": (y0 + y1) / 2.0, "_h": max(1.0, y1 - y0)})
    items.sort(key=lambda l: l["_yc"])

    groups = []
    for it in items:
        if groups:
            g = groups[-1]
            gh = sum(m["_h"] for m in g) / len(g)
            gy = sum(m["_yc"] for m in g) / len(g)
            if abs(it["_yc"] - gy) <= 0.4 * min(gh, it["_h"]):
                g.append(it)
                continue
        groups.append([it])

    rows = []
    for g in groups:
        g.sort(key=lambda m: m["bbox"][0])
        x0 = min(m["bbox"][0] for m in g)
        y0 = min(m["bbox"][1] for m in g)
        x1 = max(m["bbox"][2] for m in g)
        y1 = max(m["bbox"][3] for m in g)
        rows.append({
            "page": page,
            "text": " ".join(m["text"].strip() for m in g),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "yc": (y0 + y1) / 2.0,
            "conf": sum(m["conf"] for m in g) / len(g),
        })
    return rows


def ocr_pages(images, split_ink=None, ink=None, progress=None):
    """
    OCR every page of one document.
    Returns {"answer_rows": [...], "teacher_rows": [...], "teacher_fraction": [..], "mean_conf": f}
    answer_rows  = the student's writing (teacher's ink removed when split_ink)
    teacher_rows = ONLY the teacher's ink (marks, ticks, comments)
    """
    split_ink = config.SEPARATE_TEACHER_INK if split_ink is None else split_ink
    ink = ink or config.TEACHER_INK
    answer_rows, teacher_rows, fractions = [], [], []
    for i, img in enumerate(images):
        img = image_utils.prepare_page(img)
        if split_ink:
            student, teacher, frac = image_utils.split_ink_layers(img, ink)
        else:
            student, teacher, frac = img, None, 0.0
        fractions.append(round(frac, 5))
        answer_rows.extend(merge_rows(ocr_image(student), page=i))
        if teacher is not None and frac >= config.MIN_TEACHER_INK_FRACTION:
            teacher_rows.extend(merge_rows(ocr_image(teacher), page=i))
        if progress:
            progress(i + 1, len(images))
    confs = [r["conf"] for r in answer_rows]
    return {
        "answer_rows": answer_rows,
        "teacher_rows": teacher_rows,
        "teacher_fraction": fractions,
        "mean_conf": (sum(confs) / len(confs)) if confs else 0.0,
    }


def ocr_printed_lines(images):
    """For printed documents (question paper / key scans): plain text lines."""
    lines = []
    for i, img in enumerate(images):
        for row in merge_rows(ocr_image(image_utils.prepare_page(img)), page=i):
            lines.append(row["text"])
    return lines


# Compatibility with your earlier scratch script (test.py)
def _pdf_to_images(pdf_path, dpi=200):
    from backend import pdf_utils
    paths = []
    for i, im in enumerate(pdf_utils.render_pdf(pdf_path, dpi)):
        fd, name = tempfile.mkstemp(suffix=".png", prefix=f"ocr_page{i}_")
        os.close(fd)
        im.save(name)
        paths.append(name)
    return paths
