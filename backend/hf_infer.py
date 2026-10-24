"""
backend/hf_infer.py -- run the fine-tuned model (base + LoRA adapter) locally.

This is what makes "evaluate with the fine-tuned model" work inside the app
without converting anything to GGUF: the 3B base model is loaded in 4-bit
(about 2-2.5 GB of VRAM) and the LoRA adapter is attached on top.
"""
import json
import os
import threading

import config

_lock = threading.Lock()
_state = {"model": None, "tok": None}


def stack_available():
    try:
        import torch, transformers, peft, bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def adapter_ready():
    return os.path.exists(os.path.join(config.ADAPTER_DIR, "adapter_config.json"))


def adapter_meta():
    p = os.path.join(config.ADAPTER_DIR, "train_meta.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load():
    if _state["model"] is not None:
        return
    if not stack_available():
        raise RuntimeError("torch / transformers / peft / bitsandbytes are not installed - see README step 4.")
    if not adapter_ready():
        raise RuntimeError("No fine-tuned adapter found. Run the fine-tuning step first.")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base = adapter_meta().get("base_model", config.FINETUNE_BASE_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, config.ADAPTER_DIR)
    model.eval()
    _state["model"], _state["tok"] = model, tok


def generate(messages, disable_adapter=False, max_new_tokens=24):
    import torch
    with _lock:
        _load()
        model, tok = _state["model"], _state["tok"]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        with torch.no_grad():
            if disable_adapter:
                with model.disable_adapter():
                    out = model.generate(**inputs, **kwargs)
            else:
                out = model.generate(**inputs, **kwargs)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
