from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps
from tqdm import tqdm

# PyTorch is optional in the source code so that the script can still run on CPU.
# For your machine, use a CUDA-enabled PyTorch installation.
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

LOGGER = logging.getLogger("handwritten_pdf")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"
}

# A4 page rendered at 300 DPI.
A4_W_PX = 2480
A4_H_PX = 3508
A4_MARGIN_PX = 60
A4_W_PT = 595.275590551
A4_H_PT = 841.88976378

JPEG_QUALITY = 95
MAX_WORKING_LONG_EDGE = 3200
GPU_SHARPEN_AMOUNT = 0.18


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def list_images(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: natural_key(path.name))


def load_image(path: Path) -> np.ndarray:
    """Load an image safely, apply EXIF orientation, and return RGB uint8."""
    with Image.open(path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image)
        rgb = pil_image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8).copy()


def rotate_landscape_to_portrait(rgb: np.ndarray) -> Tuple[np.ndarray, int]:
    """Rotate physically landscape pages to portrait orientation.

    Without OCR, this intentionally does not guess 180-degree rotation.
    """
    height, width = rgb.shape[:2]
    if width > height:
        return cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE), 90
    return rgb, 0


def resize_for_processing(rgb: np.ndarray) -> np.ndarray:
    """Avoid unnecessary memory use on very large phone images."""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= MAX_WORKING_LONG_EDGE:
        return rgb

    scale = MAX_WORKING_LONG_EDGE / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def estimate_skew_degrees(gray: np.ndarray) -> float:
    """Estimate small document skew without HoughLinesP.

    A small-angle projection-profile search is used on a downscaled image.
    This avoids the OpenCV HoughLinesP shape issue from the previous script and
    is safer for handwritten pages.
    """
    height, width = gray.shape[:2]
    scale = min(1.0, 1000.0 / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Suppress tiny noise, then emphasize dark handwritten strokes.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    dark = 255 - dark

    h, w = dark.shape
    y0, y1 = int(h * 0.05), int(h * 0.95)
    x0, x1 = int(w * 0.05), int(w * 0.95)
    dark = dark[y0:y1, x0:x1]

    # Reduce the search size further for speed.
    if max(dark.shape) > 900:
        s = 900.0 / max(dark.shape)
        dark = cv2.resize(dark, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # A light threshold keeps text/line structure while ignoring the paper.
    threshold = float(np.percentile(dark, 80))
    binary = (dark > max(20.0, threshold * 0.75)).astype(np.uint8) * 255

    best_angle = 0.0
    best_score = -1.0

    # Only correct small physical page skew.  0.5-degree steps are enough.
    for angle in np.arange(-4.0, 4.01, 0.5):
        if abs(float(angle)) < 1e-9:
            rotated = binary
        else:
            h2, w2 = binary.shape
            center = (w2 / 2.0, h2 / 2.0)
            matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            rotated = cv2.warpAffine(
                binary,
                matrix,
                (w2, h2),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # Horizontal continuity score: count/weight transitions per row.
        row_energy = rotated.sum(axis=1).astype(np.float64)
        if row_energy.size < 10:
            continue

        # Text lines create repeated horizontal energy bands.
        score = float(np.mean(np.sort(row_energy)[-max(3, row_energy.size // 10):]))
        score /= float(row_energy.mean() + 1e-6)

        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if abs(best_angle) < 0.5:
        return 0.0
    return best_angle


def deskew(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    angle = estimate_skew_degrees(gray)

    if abs(angle) < 0.5:
        return rgb

    height, width = rgb.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int(round((height * sin_value) + (width * cos_value)))
    new_height = int(round((height * cos_value) + (width * sin_value)))

    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]

    return cv2.warpAffine(
        rgb,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def enhance_document(rgb: np.ndarray) -> np.ndarray:
    """Improve handwriting visibility while keeping a natural page appearance."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Mild local contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Gentle illumination/background normalization.
    background = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=21, sigmaY=21)
    l_float = l_channel.astype(np.float32)
    bg_float = np.maximum(background.astype(np.float32), 1.0)
    normalized = np.clip((l_float / bg_float) * 210.0, 0.0, 255.0).astype(np.uint8)

    # Blend with the CLAHE result to avoid making pencil/pen strokes too harsh.
    l_channel = cv2.addWeighted(l_channel, 0.70, normalized, 0.30, 0.0)

    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2RGB,
    )
    return enhanced


def fit_to_a4(rgb: np.ndarray) -> np.ndarray:
    target_w = A4_W_PX - 2 * A4_MARGIN_PX
    target_h = A4_H_PX - 2 * A4_MARGIN_PX

    height, width = rgb.shape[:2]
    scale = min(target_w / float(width), target_h / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        rgb,
        (new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4,
    )

    canvas = np.full((A4_H_PX, A4_W_PX, 3), 255, dtype=np.uint8)
    left = (A4_W_PX - new_width) // 2
    top = (A4_H_PX - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def cpu_sharpen(rgb: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    output = cv2.addWeighted(rgb, 1.0 + GPU_SHARPEN_AMOUNT, blur, -GPU_SHARPEN_AMOUNT, 0)
    return np.clip(output, 0, 255).astype(np.uint8)


def _gpu_sharpen_batch_cuda(batch_rgb: Sequence[np.ndarray]) -> list[np.ndarray]:
    assert torch is not None and F is not None

    array = np.stack(batch_rgb, axis=0)
    tensor = (
        torch.from_numpy(array)
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .to("cuda", non_blocking=True)
        / 255.0
    )

    # 5x5 Gaussian-like kernel, depthwise over RGB channels.
    kernel = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)

    with torch.inference_mode():
        blurred = F.conv2d(tensor, kernel, padding=2, groups=3)
        sharpened = torch.clamp(
            tensor + GPU_SHARPEN_AMOUNT * (tensor - blurred),
            0.0,
            1.0,
        )
        output = (
            (sharpened * 255.0)
            .round()
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

    return [image.copy() for image in output]


def enhance_gpu_or_cpu(batch_rgb: Sequence[np.ndarray]) -> list[np.ndarray]:
    if not batch_rgb:
        return []

    if torch is None or F is None or not torch.cuda.is_available():
        return [cpu_sharpen(image) for image in batch_rgb]

    try:
        return _gpu_sharpen_batch_cuda(batch_rgb)
    except RuntimeError as exc:
        # 4 GB GPUs can occasionally run out of VRAM for unusual page sizes.
        # Retry one page at a time instead of aborting the whole dataset.
        if "out of memory" not in str(exc).casefold():
            raise
        LOGGER.warning("CUDA out of memory for batch=%d; retrying one page at a time.", len(batch_rgb))
        torch.cuda.empty_cache()
        return [_gpu_sharpen_batch_cuda([image])[0] for image in batch_rgb]


def encode_jpeg(rgb: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [
            int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY,
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
        ],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode a processed page as JPEG")
    return encoded.tobytes()


def write_pdf_pages(pdf: pymupdf.Document, pages: Iterable[np.ndarray]) -> int:
    count = 0
    page_rect = pymupdf.Rect(0, 0, A4_W_PT, A4_H_PT)

    for rgb in pages:
        jpeg_bytes = encode_jpeg(rgb)
        page = pdf.new_page(width=A4_W_PT, height=A4_H_PT)
        page.insert_image(page_rect, stream=jpeg_bytes, keep_proportion=False)
        count += 1

    return count


def process_folder(folder: Path, output_pdf: Path, gpu_batch: int, do_deskew: bool) -> Tuple[int, int]:
    image_paths = list_images(folder)
    if not image_paths:
        return 0, 0

    LOGGER.info(
        "Processing %s -> %s (%d images)",
        folder,
        output_pdf,
        len(image_paths),
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    temp_pdf = output_pdf.with_name(output_pdf.stem + ".tmp.pdf")

    if temp_pdf.exists():
        try:
            temp_pdf.unlink()
        except OSError:
            pass

    pdf = pymupdf.open()
    processed_count = 0
    failed_count = 0
    prepared_batch: list[np.ndarray] = []

    try:
        def flush_batch() -> None:
            nonlocal prepared_batch, processed_count
            if not prepared_batch:
                return
            enhanced_pages = enhance_gpu_or_cpu(prepared_batch)
            processed_count += write_pdf_pages(pdf, enhanced_pages)
            prepared_batch = []

        for image_path in tqdm(image_paths, desc=folder.name, unit="img"):
            try:
                rgb = load_image(image_path)
                rgb = resize_for_processing(rgb)
                rgb, rotation = rotate_landscape_to_portrait(rgb)
                if rotation:
                    LOGGER.debug("Rotated %s by %d degrees for portrait output", image_path.name, rotation)

                if do_deskew:
                    rgb = deskew(rgb)

                rgb = enhance_document(rgb)
                rgb = fit_to_a4(rgb)
                prepared_batch.append(rgb)

                if len(prepared_batch) >= gpu_batch:
                    flush_batch()

            except Exception as exc:
                failed_count += 1
                LOGGER.exception("Failed to process %s: %s", image_path, exc)

        flush_batch()

        if processed_count == 0:
            LOGGER.error("No pages were successfully processed for %s; no PDF created.", folder)
            return 0, failed_count

        # Save only after at least one valid page exists. This prevents half-written PDFs.
        pdf.save(str(temp_pdf), garbage=4, deflate=True, clean=True)
        pdf.close()
        pdf = None  # type: ignore[assignment]

        os.replace(temp_pdf, output_pdf)
        LOGGER.info(
            "Created: %s | pages=%d | failed=%d",
            output_pdf,
            processed_count,
            failed_count,
        )
        return processed_count, failed_count

    finally:
        if pdf is not None:
            pdf.close()
        if temp_pdf.exists():
            try:
                temp_pdf.unlink()
            except OSError:
                pass


def discover_image_folders(input_root: Path, output_root: Path) -> list[Path]:
    folders: list[Path] = []
    output_root = output_root.resolve()

    for dirpath, dirnames, filenames in os.walk(input_root):
        current = Path(dirpath).resolve()

        # If the output folder is inside the input tree, do not scan it.
        try:
            current.relative_to(output_root)
            dirnames[:] = []
            continue
        except ValueError:
            pass

        if any(Path(name).suffix.casefold() in IMAGE_EXTENSIONS for name in filenames):
            folders.append(current)

    folders.sort(key=lambda path: natural_key(str(path.relative_to(input_root.resolve()))))
    return folders


def gpu_status() -> tuple[bool, str]:
    if torch is None:
        return False, "PyTorch is not installed in this Python environment."
    try:
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, f"PyTorch {torch.__version__} is installed, but CUDA is not available."
    except Exception as exc:
        return False, f"CUDA check failed: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively combine image folders into high-quality portrait A4 PDFs."
    )
    parser.add_argument("--input", type=Path, help="Input root containing folders of images")
    parser.add_argument("--output", type=Path, help="Output directory for PDFs")
    parser.add_argument(
        "--gpu-batch",
        type=int,
        default=1,
        help="Number of A4 pages processed by the GPU at once (default: 1; use 2 if VRAM allows)",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable small-angle deskew if you want maximum speed",
    )
    return parser.parse_args()


def clean_path_input(value: str) -> Path:
    return Path(value.strip().strip('"').strip("'"))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    input_root = args.input
    output_root = args.output

    if input_root is None:
        input_root = clean_path_input(input("Enter input root folder: "))
    if output_root is None:
        output_root = clean_path_input(input("Enter output folder: "))

    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    gpu_batch = max(1, int(args.gpu_batch))

    if not input_root.is_dir():
        LOGGER.error("Input folder does not exist or is not a folder: %s", input_root)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    gpu_enabled, gpu_message = gpu_status()
    LOGGER.info("Input : %s", input_root)
    LOGGER.info("Output: %s", output_root)
    LOGGER.info("%s", f"CUDA GPU enabled: {gpu_message}" if gpu_enabled else gpu_message)
    LOGGER.info("Tesseract: NOT USED")
    LOGGER.info("Orientation: EXIF + landscape-to-portrait; no OCR-based 180-degree detection")
    LOGGER.info("Deskew: %s", "enabled" if not args.no_deskew else "disabled")
    LOGGER.info("GPU batch: %d", gpu_batch)

    folders = discover_image_folders(input_root, output_root)
    if not folders:
        LOGGER.error("No image-containing folders were found under: %s", input_root)
        return 1

    LOGGER.info("Found %d image-containing folder(s).", len(folders))

    total_pages = 0
    total_failed = 0
    successful_pdfs = 0

    for folder in folders:
        relative_folder = folder.relative_to(input_root)
        # In the usual s1/s2/... layout this produces s1.pdf, s2.pdf, etc.
        output_pdf = output_root / relative_folder.parent / f"{relative_folder.name}.pdf"

        pages, failed = process_folder(
            folder,
            output_pdf,
            gpu_batch=gpu_batch,
            do_deskew=not args.no_deskew,
        )
        total_pages += pages
        total_failed += failed
        if pages > 0:
            successful_pdfs += 1

    LOGGER.info("============================================================")
    LOGGER.info("Completed")
    LOGGER.info("PDFs created : %d", successful_pdfs)
    LOGGER.info("Pages        : %d", total_pages)
    LOGGER.info("Failed images: %d", total_failed)
    LOGGER.info("Output folder: %s", output_root)
    LOGGER.info("============================================================")

    return 0 if successful_pdfs > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
