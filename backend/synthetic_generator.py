"""
backend/synthetic_generator.py -- Synthetic Error Augmentations and Data Generation.

Injects controlled OCR substitutions and corruptions to test pipeline resilience:
- Character substitutions: 1 <-> l, 0 <-> O, - <-> +, [1, 0] <-> [1, 10]
- Noise injections at 1%, 5%, and 10% CER.
- Concise vs Verbose padding generation.
"""
import random
import re
from typing import List, Dict, Tuple, Any

SUBSTITUTIONS = {
    "1": "l",
    "l": "1",
    "0": "O",
    "O": "0",
    "-": "+",
    "+": "-",
    ",": ".",
    ".": ",",
    "[1, 0]": "[1, 10]",
    "Softmax": "Soimax",
}

def inject_synthetic_ocr_errors(text: str, target_cer: float = 0.05, seed: int = 42) -> str:
    """Injects synthetic character substitutions into text to simulate OCR degradation."""
    random.seed(seed)
    chars = list(text)
    n_corruptions = int(round(len(chars) * target_cer))
    if n_corruptions == 0 and target_cer > 0:
        n_corruptions = 1

    indices = list(range(len(chars)))
    random.shuffle(indices)

    corrupted_count = 0
    for idx in indices:
        if corrupted_count >= n_corruptions:
            break
        c = chars[idx]
        if c in SUBSTITUTIONS:
            chars[idx] = SUBSTITUTIONS[c]
            corrupted_count += 1
        elif c.isalpha() and random.random() < 0.3:
            # Random typo
            chars[idx] = chr(ord('a') + random.randint(0, 25))
            corrupted_count += 1

    return "".join(chars)

def make_verbose_padded_answer(concise_answer: str) -> str:
    """Wraps a concise correct answer in verbose boilerplate to test length bias."""
    padding_prefixes = [
        "In the context of modern generative models and advanced deep learning frameworks, ",
        "To thoroughly address this inquiry from foundational theoretical principles, ",
        "Regarding the specific architectural mechanisms established in recent examination literature, "
    ]
    padding_suffixes = [
        " thereby illustrating the comprehensive theoretical underpinnings of this phenomenon.",
        " which demonstrates full adherence to the expected technical specification.",
        " as clearly proven across multiple state-of-the-art transformer implementations."
    ]
    return f"{random.choice(padding_prefixes)}{concise_answer}{random.choice(padding_suffixes)}"

def generate_synthetic_variations(text: str, n_variations: int = 3, cer_levels: List[float] = [0.01, 0.05, 0.10]) -> List[Dict[str, Any]]:
    """Generates synthetic noisy and padded variations of student answers."""
    variations = []
    for i in range(n_variations):
        cer = cer_levels[i % len(cer_levels)]
        noisy = inject_synthetic_ocr_errors(text, target_cer=cer, seed=42 + i)
        padded = make_verbose_padded_answer(text)
        variations.append({
            "variation_index": i,
            "target_cer": cer,
            "noisy_text": noisy,
            "padded_text": padded
        })
    return variations
