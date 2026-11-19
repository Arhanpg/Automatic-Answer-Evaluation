"""GPU-only TrOCR smoke test and evidence logger for Phase 4."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_phase4"
SAMPLES = OUT / "gpu_samples"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only Phase 4 run refused: torch.cuda.is_available() is False")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    model_name = "microsoft/trocr-small-handwritten"

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()

    source = ROOT / "research_phase3" / "answer_region_images"
    image_paths = sorted(source.glob("s*_page_001.png"))
    selected = image_paths[:8]
    rows = []

    torch.cuda.reset_peak_memory_stats()
    for image_path in selected:
        image = cv2.imread(str(image_path))
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGB")
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                pixel_values,
                max_length=128,
                return_dict_in_generate=True,
                output_scores=True,
            )
        torch.cuda.synchronize()
        latency = time.perf_counter() - started
        text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()

        if generated.scores:
            stacked = torch.stack(generated.scores, dim=1)
            probs = torch.softmax(stacked, dim=-1).max(dim=-1).values
            confidence = float(probs.mean().item())
            confidence_method = "mean generated-token max softmax probability"
        else:
            confidence = 0.0
            confidence_method = "scores unavailable"

        rows.append({
            "student_id": image_path.name.split("_page_", 1)[0],
            "source_image": str(image_path.resolve()),
            "source_image_sha256": sha256(image_path),
            "model_identity": model_name,
            "revision_source": "Hugging Face cached checkout",
            "device": str(device),
            "gpu_name": gpu_name,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "execution_status": "executed_on_gpu",
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "confidence": confidence,
            "confidence_method": confidence_method,
            "latency_seconds": round(latency, 4),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2),
            "cpu_fallback": False,
        })
        torch.cuda.reset_peak_memory_stats()

    fields = list(rows[0])
    with (OUT / "gpu_trocr_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(json.dumps({
        "status": "executed_on_gpu",
        "gpu": gpu_name,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "samples": len(rows),
        "nvidia_smi": nvidia.stdout.strip(),
    }, indent=2))


if __name__ == "__main__":
    main()
