"""
research_v4/model_comparison/src/feature_extractors/bm25.py

Standard BM25Okapi Implementation (Robertson et al.):
Fitted strictly on training-fold student documents and rubric texts.
Never sees or fits vocabulary/IDF statistics on validation or test folds.
"""

import math
from typing import List, Dict, Set
from collections import Counter


class BM25Okapi:
    """
    Standard Okapi BM25 implementation.
    Parameters:
      k1: term frequency saturation parameter (default: 1.5)
      b: document length normalization parameter (default: 0.75)
      epsilon: floor for negative or near-zero IDF terms (default: 0.25)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []

    def fit(self, corpus: List[List[str]]) -> "BM25Okapi":
        """
        Fits BM25 statistics (document frequency, IDF, average document length)
        strictly on the provided training corpus tokens.
        """
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            self.avgdl = 0.0
            return self

        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = float(sum(self.doc_lens) / self.corpus_size)

        # Count document frequencies
        df = Counter()
        for doc in corpus:
            unique_terms = set(doc)
            df.update(unique_terms)
        self.doc_freqs = dict(df)

        # Calculate Robertson IDF
        # IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1.0)
        self.idf = {}
        negative_idfs = []
        for term, n_q in self.doc_freqs.items():
            val = math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)
            self.idf[term] = val
            if val < 0:
                negative_idfs.append(term)

        # Apply floor for any negative IDF terms
        if self.idf:
            average_idf = sum(self.idf.values()) / len(self.idf)
            eps = self.epsilon * average_idf
            for term in negative_idfs:
                self.idf[term] = eps

        return self

    def score(self, query: List[str], doc: List[str]) -> float:
        """
        Computes BM25 score of a single document w.r.t a query.
        """
        if self.corpus_size == 0 or len(doc) == 0:
            return 0.0

        doc_len = len(doc)
        tf = Counter(doc)
        score = 0.0

        len_norm = 1.0 - self.b + self.b * (doc_len / max(1.0, self.avgdl))

        for q in query:
            if q not in tf:
                continue
            q_idf = self.idf.get(q, 0.0)
            freq = tf[q]
            num = freq * (self.k1 + 1.0)
            den = freq + self.k1 * len_norm
            score += q_idf * (num / den)

        return float(score)

    def score_corpus(self, query: List[str], corpus: List[List[str]]) -> List[float]:
        """Scores multiple documents against a query."""
        return [self.score(query, doc) for doc in corpus]
