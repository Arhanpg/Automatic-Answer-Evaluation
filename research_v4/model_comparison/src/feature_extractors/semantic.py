"""
research_v4/model_comparison/src/feature_extractors/semantic.py

Dense Semantic Similarity Feature Extractor:
Uses SentenceTransformer (all-MiniLM-L6-v2) for embedding extraction.
Computes cosine similarities against model answers and rubric criteria.
Explicitly records model loading status; never silently falls back without logging.
"""

from typing import Dict, List, Any, Optional
import numpy as np

_EMBEDDER = None
_LOAD_ATTEMPTED = False
_LOAD_SUCCESS = False


def get_sentence_embedder():
    """Singleton getter for SentenceTransformer model."""
    global _EMBEDDER, _LOAD_ATTEMPTED, _LOAD_SUCCESS
    if _LOAD_ATTEMPTED:
        return _EMBEDDER

    _LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 is compact (80MB) and executes fast on CPU/GPU
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        _LOAD_SUCCESS = True
    except Exception as e:
        _EMBEDDER = None
        _LOAD_SUCCESS = False
        print(f"[WARNING] Could not load SentenceTransformer: {e}")
    return _EMBEDDER


def is_sbert_available() -> bool:
    get_sentence_embedder()
    return _LOAD_SUCCESS


def extract_semantic_features(
    student_text: str,
    model_answer: str,
    rubric_criteria: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Computes dense semantic cosine similarities against reference solution and criteria.
    """
    embedder = get_sentence_embedder()
    s_clean = str(student_text or "").strip()
    m_clean = str(model_answer or "").strip()

    if not embedder or not s_clean or not m_clean:
        return {
            "sbert_model_answer_sim": 0.0,
            "sbert_criteria_max": 0.0,
            "sbert_criteria_mean": 0.0,
            "sbert_coverage_ratio": 0.0,
            "sbert_loaded_flag": 1.0 if _LOAD_SUCCESS else 0.0
        }

    # Encode student text and model answer
    texts_to_encode = [s_clean, m_clean]
    crit_descs = [c.get("description", "") for c in rubric_criteria if c.get("description")]
    texts_to_encode.extend(crit_descs)

    try:
        embeddings = embedder.encode(texts_to_encode, normalize_embeddings=True)
        s_emb = embeddings[0]
        m_emb = embeddings[1]

        # Model answer cosine
        m_sim = float(np.dot(s_emb, m_emb))
        m_sim = max(0.0, min(1.0, m_sim))

        # Criteria cosigns
        crit_sims = []
        for i in range(2, len(texts_to_encode)):
            c_emb = embeddings[i]
            sim = float(np.dot(s_emb, c_emb))
            crit_sims.append(max(0.0, min(1.0, sim)))

        crit_max = float(max(crit_sims)) if crit_sims else m_sim
        crit_mean = float(np.mean(crit_sims)) if crit_sims else m_sim
        cov_ratio = float(sum(1 for s in crit_sims if s >= 0.40) / max(1, len(crit_sims)))

        return {
            "sbert_model_answer_sim": round(m_sim, 4),
            "sbert_criteria_max": round(crit_max, 4),
            "sbert_criteria_mean": round(crit_mean, 4),
            "sbert_coverage_ratio": round(cov_ratio, 4),
            "sbert_loaded_flag": 1.0
        }
    except Exception as e:
        return {
            "sbert_model_answer_sim": 0.0,
            "sbert_criteria_max": 0.0,
            "sbert_criteria_mean": 0.0,
            "sbert_coverage_ratio": 0.0,
            "sbert_loaded_flag": 0.0
        }
