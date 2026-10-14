"""
scripts/merge_and_export.py

One-off helper: merges the LoRA adapter produced by backend/finetune.py
back into the base model's weights, so you get a normal (non-adapter)
model directory ready for GGUF conversion + Ollama.

Usage:
    python scripts/merge_and_export.py \
        --base Qwen/Qwen2.5-3B-Instruct \
        --adapter data/models/lora_adapter \
        --out data/models/merged_model

After this, convert to GGUF with llama.cpp (separate checkout, not part of
this repo):
    python /path/to/llama.cpp/convert-hf-to-gguf.py data/models/merged_model \
        --outfile data/models/model.gguf
    /path/to/llama.cpp/llama-quantize data/models/model.gguf \
        data/models/model-q4.gguf Q4_K_M

Then create an Ollama Modelfile:
    echo "FROM ./data/models/model-q4.gguf" > Modelfile
    ollama create answer-eval-tuned -f Modelfile

...and point config.OLLAMA_MODEL (or the model override field on the
Training page) at "answer-eval-tuned".
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base model name/path used for fine-tuning")
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter directory")
    parser.add_argument("--out", required=True, help="Where to write the merged model")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise SystemExit(
            "Needs transformers + peft installed (see README.md fine-tuning setup). "
            f"Import error: {e}"
        )

    print(f"Loading base model {args.base} ...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"Loading LoRA adapter from {args.adapter} ...")
    merged = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging adapter into base weights ...")
    merged = merged.merge_and_unload()

    print(f"Saving merged model to {args.out} ...")
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print("Done. Next: convert to GGUF with llama.cpp -- see this script's docstring.")


if __name__ == "__main__":
    main()
