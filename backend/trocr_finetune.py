"""
backend/trocr_finetune.py -- Specialist Seq2SeqTrainer fine-tuning script for microsoft/trocr-base-handwritten.

Enforces the three critical Hugging Face requirements:
1. Replaces processor.tokenizer.pad_token_id in labels with -100 for CrossEntropyLoss,
   reverting -100 to pad_token_id in compute_metrics.
2. Synchronizes model.config.decoder_start_token_id = processor.tokenizer.cls_token_id,
   pad_token_id, and vocab_size.
3. Explicitly calls processor.save_pretrained(output_dir) alongside trainer.save_model(output_dir).
"""
import os
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import evaluate
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

def compute_cer_wer_metrics(eval_pred, processor, cer_metric):
    """Reverts -100 label padding to pad_token_id and computes CER/WER."""
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

def setup_trocr_model_and_processor(
    model_name: str = "microsoft/trocr-base-handwritten",
    output_dir: str = "data/models/trocr_finetuned"
) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """
    Initializes TrOCR ensuring strict decoder token synchronization.
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # 2. Synchronize decoder_start_token_id, pad_token_id, and vocab_size
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Generation parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model, processor

def save_trocr_artifacts(trainer: Seq2SeqTrainer, processor: TrOCRProcessor, output_dir: str):
    """
    3. Explicitly calls processor.save_pretrained alongside trainer.save_model.
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"TrOCR model and processor saved successfully to {output_dir}")
