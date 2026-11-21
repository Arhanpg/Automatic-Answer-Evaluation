"""
research_v3/phase9_ocr_benchmarking/ocr_engines.py

Unified execution wrappers for OCR handwriting recognition engines:
1. TesseractEngine: Google Tesseract OCR with LSTM handwriting page segmentation
2. PaddleOCREngine: PaddleOCR PP-OCRv6 English textline detector and recognizer
3. TrOCREngine: Microsoft Vision Transformer for Optical Character Recognition (TrOCR-Small)
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import pytesseract


class TesseractEngine:
    def __init__(self, psm: int = 6, oem: int = 1):
        self.engine_name = "tesseract"
        self.config = f"--oem {oem} --psm {psm} -l eng"

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            text = pytesseract.image_to_string(image, config=self.config).strip()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "engine": self.engine_name,
                "text": text,
                "latency_ms": round(elapsed_ms, 2),
                "success": True,
                "error": None
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "engine": self.engine_name,
                "text": "",
                "latency_ms": round(elapsed_ms, 2),
                "success": False,
                "error": str(e)
            }


class PaddleOCREngine:
    def __init__(self):
        self.engine_name = "paddleocr"
        from paddleocr import PaddleOCR
        # Initialize PaddleOCR with textline orientation classification
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            img_arr = np.array(image.convert("RGB"))
            result = self.ocr.ocr(img_arr, cls=True)

            lines = []
            if result and len(result) > 0 and result[0] is not None:
                for line in result[0]:
                    if line and len(line) >= 2 and line[1] and len(line[1]) >= 1:
                        txt = line[1][0]
                        if txt:
                            lines.append(str(txt))

            full_text = " ".join(lines).strip()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "engine": self.engine_name,
                "text": full_text,
                "latency_ms": round(elapsed_ms, 2),
                "success": True,
                "error": None
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "engine": self.engine_name,
                "text": "",
                "latency_ms": round(elapsed_ms, 2),
                "success": False,
                "error": str(e)
            }


class TrOCREngine:
    def __init__(self, model_name: str = "microsoft/trocr-small-handwritten", device: Optional[str] = None):
        self.engine_name = "trocr"
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            import torch
            rgb_image = image.convert("RGB")
            # For large answer crop regions, resize height to 384 preserving aspect ratio up to max width
            w, h = rgb_image.size
            target_h = 384
            target_w = max(384, int(w * (target_h / max(1, h))))
            # TrOCR works best on line-level or paragraph-level crops up to ~1024 width
            resized_img = rgb_image.resize((min(target_w, 1024), target_h), Image.Resampling.LANCZOS)

            pixel_values = self.processor(resized_img, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_new_tokens=128, num_beams=2)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "engine": self.engine_name,
                "text": text,
                "latency_ms": round(elapsed_ms, 2),
                "success": True,
                "error": None
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "engine": self.engine_name,
                "text": "",
                "latency_ms": round(elapsed_ms, 2),
                "success": False,
                "error": str(e)
            }
