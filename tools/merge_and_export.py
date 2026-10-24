"""
OPTIONAL. The app evaluates the fine-tuned model directly (base model + LoRA adapter, see
backend/hf_infer.py), so you do NOT need this. Use it only if you also want to serve the
fine-tuned model from Ollama: merge the adapter into the base weights, then convert to GGUF
with llama.cpp and `ollama create` it.

    python tools/merge_and_export.py --base Qwen/Qwen2.5-3B-Instruct \
        --adapter data/models/lora_adapter --out data/models/merged_model
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.base), a.adapter).merge_and_unload()
model.save_pretrained(a.out); AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print("Merged model saved to", a.out, "- next: llama.cpp convert-hf-to-gguf.py, then `ollama create`.")
