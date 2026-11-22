"""
research_v4/model_comparison/src/feature_extractors/manager.py

Unified Feature Extraction Manager:
1. Coordinates all 5 feature families: Lexical, BM25/TF-IDF, Dense Semantic, Symbolic Math, Structural.
2. Fits corpus statistics (TF-IDF vectorizer, BM25Okapi index) strictly on training records.
3. Provides 11 pre-registered ablation group feature masks.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .bm25 import BM25Okapi
from .lexical import extract_lexical_features
from .semantic import extract_semantic_features
from .symbolic import extract_symbolic_features
from .structural import extract_structural_features
from ..text_normalization import clean_tokens


FEATURE_FAMILIES = {
    "lexical": [
        "lex_exact_match", "lex_char_levenshtein", "lex_token_levenshtein",
        "lex_token_jaccard", "lex_token_precision", "lex_token_recall",
        "lex_token_f1", "lex_bigram_f1", "lex_trigram_f1",
        "lex_rouge_l_f1", "lex_rouge_l_precision", "lex_rouge_l_recall",
        "lex_length_ratio"
    ],
    "bm25_tfidf": [
        "tfidf_cosine_sim", "bm25_score_vs_model_ans", "bm25_score_vs_train_corpus_max"
    ],
    "semantic": [
        "sbert_model_answer_sim", "sbert_criteria_max", "sbert_criteria_mean", "sbert_coverage_ratio"
    ],
    "symbolic": [
        "sym_raw_score", "sym_normalized_score", "sym_has_math_syntax",
        "sym_math_symbol_count", "sym_contradiction_flag"
    ],
    "structural": [
        "struct_char_count", "struct_word_count", "struct_line_count",
        "struct_equation_density", "struct_numeric_density", "struct_uppercase_ratio"
    ],
    "context": [
        "ctx_question_max_marks", "ctx_is_math_question"
    ]
}


class FeatureExtractionPipeline:
    """
    Manages fit-on-train and transform-on-test feature extraction.
    Ensures zero target leakage and zero test corpus contamination.
    """

    def __init__(self, input_modality: str = "trocr_actual"):
        self.input_modality = input_modality
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.feature_names: List[str] = []

    def fit(self, train_records: List[Dict[str, Any]]) -> "FeatureExtractionPipeline":
        """
        Fits TF-IDF and BM25 vocabularies strictly on the training fold documents and model answers.
        """
        corpus_texts = []
        tokenized_corpus = []

        for r in train_records:
            txt = str(r.get(self.input_modality, "")).strip()
            ans = str(r.get("model_answer", "")).strip()
            if txt:
                corpus_texts.append(txt)
                tokenized_corpus.append(clean_tokens(txt))
            if ans and ans not in corpus_texts:
                corpus_texts.append(ans)
                tokenized_corpus.append(clean_tokens(ans))

        # Fit TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=1000)
        if corpus_texts:
            self.tfidf_vectorizer.fit(corpus_texts)

        # Fit BM25
        self.bm25_index = BM25Okapi(k1=1.5, b=0.75)
        if tokenized_corpus:
            self.bm25_index.fit(tokenized_corpus)

        return self

    def extract_record_features(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Extracts complete feature dictionary for a single response record."""
        student_text = str(record.get(self.input_modality, "")).strip()
        model_ans = str(record.get("model_answer", "")).strip()
        subqs_key = f"subquestions_{self.input_modality.split('_')[0]}"
        subqs = record.get(subqs_key, record.get("subquestions_corrected", {}))
        qid = record.get("question_id", "1")
        max_m = record.get("max_marks", 8.0)
        criteria = record.get("rubric_criteria", [])

        feats = {}

        # 1. Lexical features
        lex_f = extract_lexical_features(student_text, model_ans)
        feats.update(lex_f)

        # 2. BM25 / TF-IDF features
        s_toks = clean_tokens(student_text)
        m_toks = clean_tokens(model_ans)

        # TF-IDF cosine
        tfidf_sim = 0.0
        if self.tfidf_vectorizer and student_text and model_ans:
            try:
                vecs = self.tfidf_vectorizer.transform([student_text, model_ans])
                tfidf_sim = float((vecs[0] * vecs[1].T).toarray()[0][0])
                tfidf_sim = max(0.0, min(1.0, tfidf_sim))
            except Exception:
                tfidf_sim = 0.0

        # BM25 score
        bm25_m = 0.0
        if self.bm25_index and s_toks and m_toks:
            bm25_m = float(self.bm25_index.score(s_toks, m_toks))

        feats["tfidf_cosine_sim"] = round(tfidf_sim, 4)
        feats["bm25_score_vs_model_ans"] = round(bm25_m, 4)
        feats["bm25_score_vs_train_corpus_max"] = round(bm25_m, 4)

        # 3. Dense Semantic features
        sem_f = extract_semantic_features(student_text, model_ans, criteria)
        feats.update(sem_f)

        # 4. Symbolic Math features
        sym_f = extract_symbolic_features(qid, student_text, subqs, max_m)
        feats.update(sym_f)

        # 5. Structural features
        struct_f = extract_structural_features(student_text)
        feats.update(struct_f)

        # 6. Context features
        feats["ctx_question_max_marks"] = float(max_m)
        feats["ctx_is_math_question"] = float(record.get("is_math_question", 0.0))

        return feats

    def transform(
        self,
        records: List[Dict[str, Any]],
        family_filter: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transforms a list of records into a feature matrix X with feature names.
        Supports ablation masking by family_filter.
        """
        if not records:
            return np.empty((0, 0)), []

        # Determine active feature names
        active_names = []
        for fam, names in FEATURE_FAMILIES.items():
            if family_filter is None or family_filter == "full":
                active_names.extend(names)
            elif family_filter == "lexical_only" and fam in ["lexical", "context"]:
                active_names.extend(names)
            elif family_filter == "bm25_tfidf_only" and fam in ["bm25_tfidf", "context"]:
                active_names.extend(names)
            elif family_filter == "dense_semantic_only" and fam in ["semantic", "context"]:
                active_names.extend(names)
            elif family_filter == "symbolic_math_only" and fam in ["symbolic", "context"]:
                active_names.extend(names)
            elif family_filter == "structural_length_only" and fam in ["structural", "context"]:
                active_names.extend(names)
            elif family_filter == "lexical_plus_semantic" and fam in ["lexical", "semantic", "context"]:
                active_names.extend(names)
            elif family_filter == "lexical_plus_symbolic" and fam in ["lexical", "symbolic", "context"]:
                active_names.extend(names)
            elif family_filter == "semantic_plus_symbolic" and fam in ["semantic", "symbolic", "context"]:
                active_names.extend(names)
            elif family_filter == "minus_lexical" and fam != "lexical":
                active_names.extend(names)
            elif family_filter == "minus_bm25" and fam != "bm25_tfidf":
                active_names.extend(names)
            elif family_filter == "minus_semantic" and fam != "semantic":
                active_names.extend(names)
            elif family_filter == "minus_symbolic" and fam != "symbolic":
                active_names.extend(names)
            elif family_filter == "minus_structural" and fam != "structural":
                active_names.extend(names)

        rows = []
        for r in records:
            f_dict = self.extract_record_features(r)
            row = [f_dict.get(fname, 0.0) for fname in active_names]
            rows.append(row)

        X = np.array(rows, dtype=np.float32)
        # Impute any NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, active_names
