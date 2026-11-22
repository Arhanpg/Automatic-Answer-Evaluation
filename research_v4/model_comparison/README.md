# Paper 2: Beyond a Universal Grader

## Model Training, Feature Engineering, and Question-Conditional Performance for Handwritten Technical Examination Assessment

**Authors**: Arhan Ghosarwade and Abhinav Gumgol  
**Affiliation**: Department of Computer Science and Engineering, Indian Institute of Information Technology Dharwad (IIIT Dharwad), Karnataka, India  
**Target Venue**: IEEE Transactions on Learning Technologies (IEEE TLT) / ACM Transactions on Computer-Human Interaction (TOCHI) / AIED  
**Workspace**: `research_v4/model_comparison/`  
**Git Branch**: `research/paper2-model-comparison`

---

## 1. Executive Summary

This research study constitutes the second independent paper in our institutional automated examination assessment research program. While Paper 1 focused on upstream layout perception, scan orientation rectification, isolated OCR benchmarking, and stage-wise error decomposition, Paper 2 addresses the downstream evaluation core:

> **Central Research Question**: Do lexical, semantic, symbolic, classical machine-learning, deep-learning, retrieval-augmented, prompted-LLM, and parameter-efficient fine-tuning approaches exhibit the same grading performance across different technical question types, answer characteristics, and input representations, or are their strengths conditional on the task and evaluation constraints?

### Core Hypothesis to Test (Not Assume)
There is no universally dominant grading model. Different model families and feature combinations may perform differently across:
1. **Conceptual prose expositions** (Question 1: BPE tokenization, vocabulary trade-offs, language modeling);
2. **Multi-step mathematical derivations** (Question 2: scaled dot-product self-attention matrix equations);
3. **Discrete decoding & algorithmic decisions** (Question 3: greedy vs. beam search vs. nucleus sampling);
4. **Statistical language model calculations** (Question 4: n-gram estimation and add-$k$ Laplace smoothing);
5. **Input representation fidelity** (human-corrected reference text vs. authentic automated TrOCR handwriting transcripts).

---

## 2. Directory Layout & Organization

```text
research_v4/model_comparison/
  README.md                           <- This master document
  protocol/
    preregistration.md                <- Locked experimental protocol, hypotheses & statistical plans
    research_questions.yaml           <- Structured YAML definitions of RQ1 through RQ7
    experiment_config.yaml            <- Model parameters, feature groups, cross-validation setup
    evidence_policy.md                <- Anti-leakage, evidence classification & validation policy
    claim_to_evidence_matrix.csv      <- Formal mapping from claims to verifiable disk artifacts
  audit/
    repository_audit.md               <- Exhaustive source-level audit of legacy code and methodological defects
    dataset_provenance.csv            <- Provenance of 37 booklets, 148 parent questions, 407 sub-answers
    annotation_provenance.csv         <- Verification of teacher marks and double-blind transcriptions
    model_reproducibility_matrix.csv  <- Audited implementation, execution, and reproducibility status of models
    input_modality_matrix.csv         <- Real vs. simulated text sources in prior benchmarks
    split_exposure_audit.json         <- Exposure audit of historical test students (s7, s14, s24, s29, s30, s37)
    privacy_audit.md                  <- Protection of student privacy and data governance protocol
  src/
    data_loader.py                    <- Leakage-safe grouped student dataset loader
    provenance.py                     <- Cryptographic hashing and artifact verification
    splits.py                         <- 5-fold GroupKFold by student ID
    text_normalization.py             <- Math-preserving normalization and exact LCS ROUGE-L
    feature_extractors/
      bm25.py                         <- True standard BM25Okapi implementation (fit on train folds only)
      lexical.py                      <- Character & token Levenshtein, exact match, token Jaccard, token F1
      semantic.py                     <- SentenceTransformer (all-MiniLM-L6-v2) cosine embeddings
      symbolic.py                     <- SymPy symbolic algebraic and numeric verification
      structural.py                   <- Surface lengths, equation density, formula indicators
      manager.py                      <- 11 feature family ablation pipeline
    model_adapters/
      baselines.py                    <- Mean, median, rubric coverage, lexical calibrator
      classical.py                    <- Ridge, ElasticNet, SVR, Random Forest, ExtraTrees, GBDT, CatBoost, LightGBM, XGBoost
      neural.py                       <- PyTorch Feature-MLP and PyTorch Text-BiLSTM with genuine forward/backward loops
      rag.py                          <- Pure training-fold hybrid BM25 + dense cosine exemplar retriever
      prompted_llm.py                 <- Local Ollama LLM adapter (llama3.2:3b, qwen2.5:7b) with strict schema parsing
      ensemble.py                     <- Out-of-fold validation-tuned stacking and Pareto selection
    training/
      cross_validation.py             <- Nested grouped CV runner (5-fold outer, 3-fold inner)
    evaluation/
      metrics.py                      <- Per-question & total QWK, MAE, RMSE, Pearson r, Spearman rho
    statistics/
      bootstrap.py                    <- 2,000 cluster-bootstrap resamples by student ID & paired permutation tests
    reporting/
      table_generator.py              <- Generates markdown/LaTeX summary tables
  tests/                              <- Comprehensive anti-leakage, metric & reproducibility tests
  runs/                               <- JSONL predictions, run telemetry, and execution metadata
  reports/
    FINAL_MODEL_COMPARISON_REPORT.md  <- Complete scientific analysis of all models & feature families
    EXPERIMENT_RUNBOOK.md             <- Step-by-step reproduction guide
    PAPER2_OUTLINE.md                 <- Manuscript structure and supported claim taxonomy
  figures/                            <- Performance curves, heatmaps, Pareto scatterplots, ablations
  tables/                             <- Machine-readable CSV results and LaTeX tables
```

---

## 3. Strict Non-Negotiable Research Integrity Rules

1. **Zero Target Leakage**: Absolutely no feature, scaler, vectorizer, BM25 vocabulary, or model parameter may read or depend on test labels (`y_test`, teacher marks, or absolute errors).
2. **Zero Simulation**: Every score, metric, loss, latency, and prediction must be produced by live, deterministic code execution on disk.
3. **Cluster-Aware Statistics**: Every statistical inference, confidence interval, and permutation test clusters by student ($N=37$), carrying all 4 parent-question responses together.
4. **Clear Evidence Classification**: Every reported result is explicitly classified as `VERIFIED_LIVE`, `VERIFIED_EXISTING_ARTIFACT`, `PROVISIONAL`, `EXPLORATORY`, `SYNTHETIC`, or `NOT_RUN`.
