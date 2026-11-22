"""
research_v4/model_comparison/src/feature_extractors/structural.py

Surface Structure and Answer Style Feature Extractor:
1. Character length, token count, sentence/line count.
2. Equation density and numeric token density.
3. Uppercase ratio and symbol ratio.
"""

import re
from typing import Dict, List


def extract_structural_features(student_text: str) -> Dict[str, float]:
    """
    Computes length, formatting, and mathematical token density.
    """
    s_raw = str(student_text or "")
    char_len = len(s_raw)
    words = s_raw.split()
    word_cnt = len(words)
    lines = [l for l in s_raw.splitlines() if l.strip()]
    line_cnt = len(lines)

    if char_len == 0 or word_cnt == 0:
        return {
            "struct_char_count": 0.0,
            "struct_word_count": 0.0,
            "struct_line_count": 0.0,
            "struct_equation_density": 0.0,
            "struct_numeric_density": 0.0,
            "struct_uppercase_ratio": 0.0
        }

    # Math operators and equations
    eq_chars = sum(1 for c in s_raw if c in "+-*/^=<>_[]{}()\\%")
    eq_density = float(eq_chars / char_len)

    # Numbers
    num_matches = re.findall(r"\b\d+(?:\.\d+)?\b", s_raw)
    num_density = float(len(num_matches) / word_cnt)

    # Uppercase
    uppers = sum(1 for c in s_raw if c.isupper())
    upper_ratio = float(uppers / max(1, sum(1 for c in s_raw if c.isalpha())))

    return {
        "struct_char_count": float(char_len),
        "struct_word_count": float(word_cnt),
        "struct_line_count": float(line_cnt),
        "struct_equation_density": round(eq_density, 4),
        "struct_numeric_density": round(num_density, 4),
        "struct_uppercase_ratio": round(upper_ratio, 4)
    }
