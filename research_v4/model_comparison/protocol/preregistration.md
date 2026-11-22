# Preregistration: Beyond a Universal Grader

## Model Training, Feature Engineering, and Question-Conditional Performance for Handwritten Technical Examination Assessment

**Date of Preregistration**: November 22, 2026  
**Authors**: Arhan Ghosarwade & Abhinav Gumgol  
**Affiliation**: Department of Computer Science & Engineering, IIIT Dharwad  
**Status**: LOCKED BEFORE TRAINING

---

## 1. Study Overview & Hypotheses

### Primary Research Question
Do lexical, semantic, symbolic, classical machine-learning, deep-learning, retrieval-augmented, prompted-LLM, and parameter-efficient fine-tuning approaches exhibit the same grading performance across different technical question types, answer characteristics, and input representations, or are their strengths conditional on the task and evaluation constraints?

### Core Hypotheses
- **H1 (Absence of Universal Dominance)**: No single model family dominates across all four examination questions. Tree-based ensembles and classical ML will excel at conceptual prose with rich feature vectors, whereas symbolic verifiers will dominate exact mathematical derivations.
- **H2 (Feature Synergy)**: Dense semantic representations (SentenceTransformer) combined with exact symbolic features (SymPy) will outperform pure lexical baselines (BM25, TF-IDF, Jaccard).
- **H3 (Representation Sensitivity)**: Model rankings will remain relatively stable, but absolute grading accuracy will degrade predictably when transitioning from clean human-adjudicated reference text to realistic TrOCR handwriting transcriptions.
- **H4 (Computational Pareto Frontier)**: Compact classical models (Ridge, LightGBM) will achieve superior accuracy-to-latency and accuracy-to-memory trade-offs compared to large prompted LLMs on institutional hardware.

---

## 2. Dataset & Unit of Analysis

- **Corpus**: 37 complete student examination booklets (195 scanned pages) in advanced Computer Science (Large Language Models examination) at IIIT Dharwad.
- **Principal Unit of Analysis**: **Student $\times$ Parent-Question Response** ($N = 148$ responses: 37 students $\times$ 4 parent questions).
  - Question 1: Max 8.0 marks (3 subquestions: 1a, 1b, 1c)
  - Question 2: Max 6.0 marks (3 subquestions: 2a, 2b, 2c)
  - Question 3: Max 3.0 marks (3 subquestions: 3a, 3b, 3c)
  - Question 4: Max 3.0 marks (2 subquestions: 4a, 4b / 4)
  - Total Exam Max Marks: 20.0
- **Grouping Unit**: The **Student** is the invariant cluster unit. All 4 parent-question responses from a given student are held together in the same split fold.

---

## 3. Cross-Validation & Splitting Protocol

- **Outer Cross-Validation**: 5-fold GroupKFold by student ID (approx. 7–8 students per test fold, 29–30 students for training).
- **Inner Tuning**: 3-fold grouped cross-validation within the training folds for hyperparameter tuning, feature selection, and ensemble weight learning.
- **Strict Anti-Leakage Constraints**:
  - Vectorizers (TF-IDF), BM25 corpus dictionaries, feature scalers (StandardScaler), and imputation thresholds must be fit strictly on training folds.
  - Zero test label leakage: no feature, prompt, or model parameter may read or depend on `y_test`.
- **Legacy Test Set Reporting**: The historical 6-booklet test cohort (`s7, s14, s24, s29, s30, s37`) will be reported only as a secondary historical exploratory partition to compare against legacy claims.

---

## 4. Evaluated Model Paradigms

1. **Non-Learning Baselines**: Mean Predictor, Median Predictor, Rubric Criterion Coverage, Lexical Score Calibrator.
2. **Deterministic Symbolic Grader**: SymPy algebraic and numerical equivalence verifier with dual-override rules.
3. **Classical Machine Learning**: Ridge Regression, ElasticNet, Lasso, Linear Regression, Support Vector Regression (SVR with RBF kernel), Random Forest, ExtraTrees, Gradient Boosting (GBDT), CatBoost, LightGBM, XGBoost.
4. **Deep Learning**:
   - PyTorch Feature-MLP (2-layer MLP with BatchNorm, Dropout, and ReLU).
   - PyTorch Text-BiLSTM (Bidirectional LSTM on word embeddings with self-attention pooling).
5. **Retrieval-Augmented Grading (RAG)**:
   - BM25-Only Retrieval.
   - Dense SBERT Cosine Retrieval.
   - Hybrid BM25 + Dense Retrieval (candidates strictly limited to training-fold students).
6. **Prompted Local LLMs**:
   - `llama3.2:3b` (Zero-shot rubric, Few-shot training exemplar, Criterion-decomposed).
   - `qwen2.5:7b` (Zero-shot rubric, JSON-constrained).
7. **Hybrid Ensembles**:
   - Validation-Tuned Stacking Regressor (Ridge meta-learner trained on out-of-fold validation predictions).
   - Pareto-Optimal Multi-Expert Ensemble.

---

## 5. Metrics & Statistical Evaluation

### Primary Endpoints
- **Quadratic Weighted Kappa (QWK)**: Ordinal agreement with teacher marks.
- **Mean Absolute Error (MAE)**: Absolute mark deviation on the question's native scale.

### Secondary Endpoints
- **Root Mean Squared Error (RMSE)**.
- **Normalized MAE (N-MAE)**: $\text{MAE} / \text{MaxMarks}$.
- **Within $\pm0.5$ Mark & Within $\pm1.0$ Mark Agreement**.
- **Spearman $\rho$ & Pearson $r$ Correlation**.
- **Inference Latency (ms)**, **Peak RAM/VRAM**, **Model Size (MB)**.

### Statistical Inference Protocol
- **Clustered Bootstrap Resampling**: $B = 2,000$ bootstrap resamples at the **student level** ($N = 37$).
- **Paired Permutation Tests**: Two-tailed paired permutation test on per-student absolute errors.
- **Multiplicity Correction**: Holm-Bonferroni correction for family-wise error rate control.
