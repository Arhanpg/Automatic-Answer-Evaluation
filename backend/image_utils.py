"""
backend/image_utils.py -- image helpers for corrected answer sheets.

split_ink_layers() is the key trick: a corrected script has the student's
handwriting (blue/black) and the teacher's marks (red) on the SAME page.
We separate them by colour so that
  * the student's answers are OCR'd with the teacher's red ink removed, and
  * the teacher's marks are OCR'd on their own, with nothing else on the page.
That gives clean text for folder 1 and clean marks for folder 2.
"""
import numpy as np
from PIL import Image

MAX_EDGE = 1400
MIN_WIDTH = 900


def prepare_page(img, max_edge=MAX_EDGE, min_width=MIN_WIDTH):
    """Downscale huge phone photos, upscale tiny scans. Colour is preserved."""
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_edge:
        s = max_edge / float(longest)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    elif w < min_width:
        s = min_width / float(w)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return img


def ink_mask(arr, ink="red"):
    """Boolean mask of pixels that look like the teacher's ink."""
    a = arr.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if ink == "green":
        return (g > 90) & (g - r > 40) & (g - b > 30)
    # red / pink / orange-red pen
    return (r > 110) & (r - g > 55) & (r - b > 45)


def _dilate(mask, radius=1):
    try:
        import cv2
        k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    except ImportError:
        out = mask.copy()
        h, w = mask.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ys, yd = (slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0)))
                xs, xd = (slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0)))
                out[yd, xd] |= mask[ys, xs]
        return out


def split_ink_layers(img, ink="red"):
    """
    Returns (student_layer, teacher_layer, teacher_fraction)
      student_layer : original page with the teacher's ink painted white
      teacher_layer : white page with ONLY the teacher's ink, drawn in black
      teacher_fraction : share of pixels that are teacher ink (0..1)
    """
    arr = np.asarray(img.convert("RGB"))
    mask = ink_mask(arr, ink)
    frac = float(mask.mean())
    if frac == 0.0:
        return img.convert("RGB"), Image.new("RGB", img.size, "white"), 0.0

    wide = _dilate(mask, 2)            # also remove the anti-aliased red fringe
    student = arr.copy()
    student[wide] = 255

    teacher = np.full_like(arr, 255)
    teacher[_dilate(mask, 1)] = 0      # thicken strokes slightly: easier to read
    return Image.fromarray(student), Image.fromarray(teacher), frac


def save_page_jpeg(img, path, max_edge=1800, quality=82):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        s = max_edge / float(max(w, h))
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img.save(path, "JPEG", quality=quality)
