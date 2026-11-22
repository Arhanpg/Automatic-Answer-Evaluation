"""
research_v4/model_comparison/src/model_adapters/rag.py

Retrieval-Augmented Grading (RAG) Model Adapters:
1. BM25OnlyRAG: Lexical retrieval using standard BM25Okapi.
2. DenseSBERTRAG: Semantic embedding retrieval using SentenceTransformer.
3. HybridRAG: Weighted BM25 + Dense cosine score fusion.

Strict Anti-Leakage Constraints:
- Candidate pool strictly restricted to training-fold students.
- Query student and test-fold students are excluded from candidate pool.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ..feature_extractors.bm25 import BM25Okapi
from ..feature_extractors.semantic import get_sentence_embedder
from ..text_normalization import clean_tokens


class RetrievalAugmentedGrader:
    """
    RAG grader using training-fold exemplar retrieval and similarity-weighted label aggregation.
    """

    def __init__(self, strategy: str = "hybrid", top_k: int = 3, input_key: str = "trocr_actual"):
        self.strategy = strategy  # "bm25", "dense", "hybrid"
        self.top_k = top_k
        self.input_key = input_key
        self.train_pool: Dict[str, List[Dict[str, Any]]] = {}
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        self.dense_embeddings: Dict[str, np.ndarray] = {}

    def fit(self, train_records: List[Dict[str, Any]]) -> "RetrievalAugmentedGrader":
        """Indexes training-fold responses partitioned by question ID."""
        embedder = get_sentence_embedder()
        self.train_pool = {"1": [], "2": [], "3": [], "4": []}

        for r in train_records:
            qid = str(r["question_id"])
            if qid in self.train_pool:
                self.train_pool[qid].append(r)

        # Build indices per question
        for qid, pool in self.train_pool.items():
            if not pool:
                continue

            # 1. BM25 index
            docs = [clean_tokens(str(r.get(self.input_key, ""))) for r in pool]
            bm25 = BM25Okapi()
            bm25.fit(docs)
            self.bm25_indices[qid] = bm25

            # 2. Dense embeddings
            if embedder:
                texts = [str(r.get(self.input_key, "")).strip() or "empty" for r in pool]
                embs = embedder.encode(texts, normalize_embeddings=True)
                self.dense_embeddings[qid] = embs

        return self

    def predict(self, test_records: List[Dict[str, Any]]) -> np.ndarray:
        """Retrieves top-k exemplars and returns weighted mark estimate."""
        embedder = get_sentence_embedder()
        preds = []

        for r in test_records:
            qid = str(r["question_id"])
            q_text = str(r.get(self.input_key, "")).strip()
            max_m = float(r["max_marks"])
            q_sid = r["student_id"]

            pool = self.train_pool.get(qid, [])
            # Filter candidate pool: strictly exclude same student
            valid_candidates = [(idx, cand) for idx, cand in enumerate(pool) if cand["student_id"] != q_sid]

            if not valid_candidates:
                preds.append(max_m * 0.5)
                continue

            scores = []
            q_toks = clean_tokens(q_text)

            # Compute BM25 scores
            bm25_scores = {}
            if self.strategy in ["bm25", "hybrid"] and qid in self.bm25_indices:
                for idx, cand in valid_candidates:
                    c_toks = clean_tokens(str(cand.get(self.input_key, "")))
                    bm25_scores[idx] = self.bm25_indices[qid].score(q_toks, c_toks)

            # Compute Dense scores
            dense_scores = {}
            if self.strategy in ["dense", "hybrid"] and embedder and qid in self.dense_embeddings:
                q_emb = embedder.encode([q_text or "empty"], normalize_embeddings=True)[0]
                for idx, cand in valid_candidates:
                    c_emb = self.dense_embeddings[qid][idx]
                    dense_scores[idx] = float(np.dot(q_emb, c_emb))

            # Combine scores
            for idx, cand in valid_candidates:
                if self.strategy == "bm25":
                    s = bm25_scores.get(idx, 0.0)
                elif self.strategy == "dense":
                    s = dense_scores.get(idx, 0.0)
                else:  # hybrid
                    b_norm = max(0.0, min(1.0, bm25_scores.get(idx, 0.0) / 10.0))
                    d_norm = max(0.0, min(1.0, dense_scores.get(idx, 0.0)))
                    s = 0.5 * b_norm + 0.5 * d_norm
                scores.append((s, cand["teacher_mark"]))

            # Sort by similarity descending
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k_items = scores[:self.top_k]

            weights = [max(1e-4, item[0]) for item in top_k_items]
            marks = [item[1] for item in top_k_items]

            weighted_mark = sum(w * m for w, m in zip(weights, marks)) / sum(weights)
            preds.append(min(max_m, max(0.0, weighted_mark)))

        return np.array(preds, dtype=np.float64)
