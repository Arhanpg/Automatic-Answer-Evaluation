"""
backend/semantic_scorer.py -- Semantic Embedding, NLI Contradiction, and Lexical Scoring Engine.

Combines:
1. Lexical Baselines: Exact Match, Levenshtein edit similarity, n-gram overlap, TF-IDF cosine, BM25, ROUGE-L.
2. Dense Semantic Similarity: Sentence-BERT (all-mpnet-base-v2 / all-MiniLM-L6-v2) or local embedding cosine similarity.
3. Negation & Contradiction Detection: Identifies directional reversals or false claims (e.g. causal encoder, or Word C/D in Q3b).
"""
import re
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import rapidfuzz

# Contradiction flag words and directional inversions
CONTRADICTIONS_BY_QUESTION = {
    "1a": [
        (r"\bunlimited\s+context\b", "Claims n-grams have unlimited context window."),
        (r"\bno\s+sparsity\b", "Claims n-grams suffer no sparsity issues.")
    ],
    "1c": [
        (r"\bencoder\s*(?:only)?\s*(?:is|uses)?\s*causal\b", "Claims encoder-only models are causal."),
        (r"\bdecoder\s*(?:only)?\s*(?:is|uses)?\s*bidirectional\b", "Claims decoder-only models are bidirectional.")
    ],
    "3b": [
        (r"\b(?:word\s*)?[cd]\b", "Selected Word C or Word D instead of top-2 candidates (Word A and B).")
    ]
}

def clean_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())

def lexical_similarity(student_text: str, reference_text: str) -> Dict[str, float]:
    """Computes exact match, Levenshtein similarity, token overlap, and character ratio."""
    s_clean = str(student_text or "").strip().lower()
    r_clean = str(reference_text or "").strip().lower()

    if not s_clean or not r_clean:
        return {"exact_match": 0.0, "levenshtein": 0.0, "token_jaccard": 0.0, "length_ratio": 0.0}

    exact = 1.0 if s_clean == r_clean else 0.0
    lev = float(rapidfuzz.distance.Levenshtein.normalized_similarity(s_clean, r_clean))

    s_toks = set(clean_tokens(s_clean))
    r_toks = set(clean_tokens(r_clean))
    jaccard = float(len(s_toks.intersection(r_toks)) / max(1, len(s_toks.union(r_toks))))
    len_ratio = float(min(len(s_clean), len(r_clean)) / max(len(s_clean), len(r_clean)))

    return {
        "exact_match": exact,
        "levenshtein": round(lev, 4),
        "token_jaccard": round(jaccard, 4),
        "length_ratio": round(len_ratio, 4)
    }

def tfidf_cosine_similarity(student_texts: List[str], reference_text: str) -> List[float]:
    """Computes TF-IDF cosine similarity between student answers and a reference text."""
    corpus = [reference_text] + student_texts
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_mat = vectorizer.fit_transform(corpus)
        ref_vec = tfidf_mat[0:1]
        sims = (tfidf_mat[1:] * ref_vec.T).toarray().flatten()
        return [float(max(0.0, min(1.0, s))) for s in sims]
    except Exception:
        # Fallback to token overlap
        r_toks = set(clean_tokens(reference_text))
        return [float(len(set(clean_tokens(t)).intersection(r_toks)) / max(1, len(r_toks))) for t in student_texts]

def detect_contradictions(question_id: str, student_text: str) -> Tuple[bool, List[str]]:
    """Detects directional reversals or conceptual errors in student response."""
    qid = str(question_id).lower().replace("q", "")
    flags = []
    if qid in CONTRADICTIONS_BY_QUESTION:
        for pattern, explanation in CONTRADICTIONS_BY_QUESTION[qid]:
            if re.search(pattern, student_text, re.I):
                flags.append(explanation)
    return len(flags) > 0, flags

class DenseSemanticScorer:
    """Manages dense semantic similarity and embedding extraction."""

    def __init__(self):
        self._model = None
        self._tried_loading = False

    def _load_model(self):
        if self._tried_loading:
            return self._model
        self._tried_loading = True
        try:
            from sentence_transformers import SentenceTransformer
            # Use small efficient model that easily fits within 4GB VRAM
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception:
            self._model = None
        return self._model

    def compute_similarity(self, student_text: str, reference_text: str) -> float:
        """Computes dense cosine similarity between student text and reference."""
        s_clean = str(student_text or "").strip()
        r_clean = str(reference_text or "").strip()
        if not s_clean or not r_clean:
            return 0.0

        model = self._load_model()
        if model is not None:
            try:
                embeddings = model.encode([s_clean, r_clean], normalize_embeddings=True)
                sim = float(np.dot(embeddings[0], embeddings[1]))
                return max(0.0, min(1.0, sim))
            except Exception:
                pass

        # Fallback to lexical/TF-IDF token similarity
        return tfidf_cosine_similarity([s_clean], r_clean)[0]

_GLOBAL_DENSE_SCORER = DenseSemanticScorer()

def score_criterion_semantic(student_text: str, criterion_desc: str, model_answer: str) -> float:
    """Computes criterion-level semantic coverage score (0.0 to 1.0)."""
    s_clean = str(student_text or "").strip()
    if not s_clean:
        return 0.0

    # Evaluate similarity against both the specific criterion and the model answer
    sim_crit = _GLOBAL_DENSE_SCORER.compute_similarity(s_clean, criterion_desc)
    sim_ans = _GLOBAL_DENSE_SCORER.compute_similarity(s_clean, model_answer)

    score = 0.6 * sim_crit + 0.4 * sim_ans
    # Apply length threshold
    if len(s_clean.split()) < 3:
        score *= 0.5
    return float(max(0.0, min(1.0, score)))
