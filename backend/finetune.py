"""
backend/finetune.py -- QLoRA fine-tuning on the TRAINING papers only.

Each training example = one question of one training paper:
    prompt  = the same prompt the grader uses (question, model answer, rubric, student answer)
    target  = {"marks": <what the teacher gave>}
Only the target tokens contribute to the loss (the prompt is masked), so the
model learns "given this answer, the teacher awards this many marks".

Hardware (RTX 3050): 4-bit base model + LoRA adapters. On 4 GB VRAM keep the
3B model (default). On 8 GB you may try a 7B model via AAE_FINETUNE_BASE_MODEL.
With ~10 papers x 4 questions you get ~40 examples: enough to adapt marking
STYLE; more verified training papers = better results.
"""
import json
import os

import config
from backend import exam_schema as es
from backend import prompts


def build_training_examples(exam, train_data, rubrics):
    """train_data: [{"roll","answers","marks"}] - TRAIN split only (caller guarantees)."""
    examples = []
    for rec in train_data:
        for q in exam["questions"]:
            marks = rec["marks"].get(q["id"])
            if marks is None:
                continue
            parts = {a: rec["answers"].get(a, "") for a in es.question_answer_ids(exam, q["id"])}
            msgs = prompts.build_messages(exam, q["id"], parts, rubrics.get(q["id"]) if rubrics else None,
                                          with_rationale=False)
            examples.append({"roll": rec["roll"], "qid": q["id"], "messages": msgs,
                             "target": prompts.target_json(marks)})
    return examples


def save_training_set(examples, path=None):
    path = path or os.path.join(config.MODELS_DIR, "training_set.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path


def run_finetune(examples, use_rubric, progress=None):
    from backend import hf_infer
    if not hf_infer.stack_available():
        raise RuntimeError(
            "Fine-tuning needs torch (CUDA build), transformers, peft, bitsandbytes, datasets, accelerate.\n"
            "Install PyTorch from https://pytorch.org/get-started/locally/ first, then:\n"
            "  pip install transformers peft bitsandbytes datasets accelerate")
    if not examples:
        raise ValueError("No training examples: the training papers have no marks yet.")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments)

    out_dir = config.ADAPTER_DIR
    os.makedirs(out_dir, exist_ok=True)
    if progress:
        progress("Loading base model", 0, 1)

    tok = AutoTokenizer.from_pretrained(config.FINETUNE_BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=config.USE_4BIT_QUANTIZATION, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(config.FINETUNE_BASE_MODEL, quantization_config=bnb,
                                                 device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

    def encode(ex):
        prompt = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(ex["target"] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        room = config.FINETUNE_MAX_TOKENS - len(r_ids)
        if len(p_ids) > room:
            p_ids = p_ids[-room:]
        ids = p_ids + r_ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + r_ids}

    ds = Dataset.from_list([{"messages": e["messages"], "target": e["target"]} for e in examples]).map(
        encode, remove_columns=["messages", "target"])

    class Report(TrainerCallback):
        last_loss = None

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                Report.last_loss = logs["loss"]
            if progress:
                msg = f"step {state.global_step}/{state.max_steps}"
                if Report.last_loss is not None:
                    msg += f"  loss {Report.last_loss:.3f}"
                progress(msg, state.global_step, max(1, state.max_steps))

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=config.FINETUNE_EPOCHS, per_device_train_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=config.FINETUNE_LR, fp16=True, logging_steps=1,
        save_strategy="no", report_to=[], remove_unused_columns=False, lr_scheduler_type="cosine",
        warmup_ratio=0.1)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[Report()])
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": config.FINETUNE_BASE_MODEL, "use_rubric": use_rubric,
                   "n_examples": len(examples), "train_papers": sorted({e["roll"] for e in examples}),
                   "epochs": config.FINETUNE_EPOCHS, "final_loss": Report.last_loss}, f, indent=2)
    return out_dir
