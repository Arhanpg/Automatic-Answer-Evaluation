# Manuscript Drafting Guide & Structural Outline: Paper 2

**Title**: Beyond a Universal Grader: Feature- and Question-Conditional Performance of Models for Handwritten Technical Examination Assessment  
**Authors**: Arhan Ghosarwade and Abhinav Gumgol  
**Affiliation**: Department of Computer Science and Engineering, Indian Institute of Information Technology (IIIT) Dharwad  
**Target Venue**: IEEE Transactions on Learning Technologies / ACM Transactions on Computing Education / Computers & Education  
**Working Directory**: `research_v4/model_comparison/`  
**Git Branch**: `research/paper2-model-comparison`  

---

## Abstract Structure (250 words target)
1. **Context**: Automated Short Answer Grading (ASAG) in handwritten technical university examinations requires assessing multimodal responses comprising conceptual prose, mathematical derivations, algorithmic traces, and statistical calculations under noisy optical character recognition (OCR).
2. **Problem**: Prior literature frequently assumes a single dominant grading paradigm (e.g., fine-tuned BERT or large language models) or evaluates models on synthetically constructed datasets, ignoring OCR-induced degradation and intra-student correlation.
3. **Methodology**: We present an exhaustive, leakage-free empirical benchmark across 37 authentic student examination booklets (148 parent question evaluations, 195 physical pages) from a senior undergraduate course on Transformers and Natural Language Processing. We evaluate 24 model configurations spanning non-learning baselines, regularized linear estimators, gradient-boosted decision trees (CatBoost, LightGBM, XGBoost), deep neural networks (PyTorch MLP, BiLSTM), retrieval-augmented generation (RAG), and prompted local LLMs (Llama-3.2:3b). All evaluations use strict 5-fold student-clustered cross-validation with clustered bootstrap 95% confidence intervals and permutation hypothesis testing.
4. **Key Results**: The "Universal Grader" hypothesis is empirically disproven. While CatBoost (MAE 1.1926, QWK 0.7824) and validation-tuned stacking (MAE 1.2128, QWK 0.7898) achieve the highest overall human agreement, model supremacy is strictly question-conditional: SVR and RAG dominate conceptual prose (MAE 1.25–1.32), CatBoost dominates attention matrix derivations (MAE 1.82), and ElasticNet dominates algorithmic traces (MAE 0.62, 86.5% within 1 mark). Deep sequence models and unprompted small LLMs suffer from severe sample-efficiency bottlenecks and scale drift (Llama-3.2 MAE 2.5845). Feature ablations reveal that structural length and exact symbolic verification provide critical anchors against OCR transcription noise.
5. **Impact**: We define the computational Pareto frontier where Ridge regression and CatBoost deliver near-human grading accuracy at sub-centisecond latencies, establishing an open-source, reproducible baseline for real-world automated examination assessment.

---

## Section-by-Section Manuscript Structure

### Section I: Introduction
- **1.1 The Challenge of High-Stakes Technical Examination Assessment**:
  - Distinguishing open-ended university technical exams from multiple-choice or short-phrase tests.
  - The multimodal nature of computer science curricula: prose explanations, matrix algebra, algorithmic execution traces, and numerical probability computations.
- **1.2 The Two-Stage Perception-Evaluation Paradigm**:
  - Contextualizing Paper 1 (Layout perception and handwriting recognition) and Paper 2 (Feature engineering, model training, and grading performance).
  - The critical interface: How OCR transcription hypotheses interact with grading models.
- **1.3 Research Questions (RQ1–RQ7)**:
  - Formulate the 7 pre-registered research questions addressing universality, feature synergy, question typology, modality sensitivity, noise robustness, Pareto efficiency, and LLM viability.
- **1.4 Summary of Contributions**:
  - First comprehensive, leakage-free empirical benchmark on authentic handwritten university exam scripts.
  - Release of 33 engineered multimodal features, 24 model adapters, and reproducible student-clustered evaluation protocols.

### Section II: Related Work & Methodological Vulnerabilities
- **2.1 Evolution of Automated Short Answer Grading (ASAG)**:
  - Phase 1: Lexical overlap and string kernels (Levenshtein, Jaccard, BLEU).
  - Phase 2: Latent semantic analysis (LSA) and word embeddings (Word2Vec, GloVe).
  - Phase 3: Pre-trained language models (BERT, RoBERTa, Sentence-BERT).
  - Phase 4: Large language models and prompted generation.
- **2.2 Methodological Flaws in Prior Automated Grading Literature**:
  - *Target Leakage*: Blending test ground-truth labels into predictions or fitting scalers on the entire dataset.
  - *Student Leakage*: Splitting by response rather than by student, causing memorization of individual writing style and handwriting quirks.
  - *Evaluation on Clean Text*: Assuming perfect digital transcription and ignoring OCR character and word error rates.
  - *Deterministic Over-Reliance*: Deploying rigid string-matching rules that collapse under minor handwriting variations.

### Section III: Dataset Provenance & Experimental Protocol
- **3.1 Institutional Corpus**:
  - 37 complete student booklets, 195 scanned physical pages, 148 parent question evaluations.
  - Detailed taxonomy of Questions 1–4 (Max marks: 8m, 10m, 8m, 4m).
- **3.2 Input Modality Triad**:
  - `corrected_transcript`: Phase 8 dual-annotated, human-adjudicated ground truth.
  - `trocr_actual`: Phase 9 fine-tuned TrOCR hypothesis.
  - `tesseract_actual`: Phase 9 preprocessed Tesseract 5.3 hypothesis.
- **3.3 Anti-Leakage Cross-Validation Architecture**:
  - 5-fold `GroupKFold` strictly grouped by student ID.
  - Verification of disjoint student sets ($0$ overlap across all folds).
  - In-fold fitting of all scalers, TF-IDF vectorizers, and BM25 vocabularies.
- **3.4 Statistical Evaluation Framework**:
  - Quadratic Weighted Kappa (QWK) on half-mark quantized scale.
  - Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Pearson $r$, Spearman $\rho$, and within-margin accuracy ($\pm 0.5$, $\pm 1.0$ mark).
  - Student-clustered bootstrap ($B=2,000$) and paired permutation testing ($M=2,000$).
  - **Evidence Links**: `audit/dataset_provenance.csv`, `protocol/preregistration.md`.

### Section IV: Feature Engineering Framework
- **4.1 Representational Taxonomy (33 Features)**:
  - *Lexical Features (15)*: Exact match, character/token Levenshtein, token Jaccard, precision/recall/F1, n-gram overlaps (1-3), true DP-based LCS ROUGE-L, length ratio.
  - *Information Retrieval & TF-IDF (2)*: In-fold fitted Robertson BM25Okapi ($k_1=1.5, b=0.75$), TF-IDF cosine similarity.
  - *Dense Semantic Features (4)*: SentenceTransformer (`all-MiniLM-L6-v2`) 384-d cosine similarity, subquestion mean similarity, L2 distance, dot product.
  - *Symbolic Mathematical Verification (4)*: SymPy algebraic expression verifier, numerical matrix value equivalence, canonical formula matcher, contradiction detection.
  - *Structural Complexity & Length (5)*: Character, word, and line counts, math token density, uppercase ratio.
- **Evidence Links**: `src/feature_extractors/`, `tests/test_feature_extractors.py`.

### Section V: Empirical Results: The Myth of the Universal Grader
- **5.1 Overall Model Benchmark**:
  - Detailed discussion of Table 1 (`tables/main_model_comparison.csv`).
  - Analysis of CatBoost (MAE 1.1926, QWK 0.7824) and Validation-Tuned Stacking (MAE 1.2128, QWK 0.7898).
  - Why the non-learning median baseline (MAE 1.1892) provides a strong error floor but poor discriminative ranking (Pearson $r=0.7537$ vs. $0.8073$).
- **5.2 The Failure of Deep Sequence Models on Small Cohorts**:
  - Why PyTorch BiLSTM (MAE 1.5845, QWK 0.5619) overfits compared to tabular ML on engineered features.
- **5.3 The Prompted LLM Performance Deficit**:
  - Empirical evaluation of Llama-3.2:3b (MAE 2.5845, QWK 0.2390, latency 7,085 ms).
  - Analysis of scale drift and lack of rubric calibration.
- **Evidence Links**: `tables/main_model_comparison.csv`, `figures/overall_model_performance_qwk_mae.png`, Claims C01, C02, C10.

### Section VI: Question-Conditional Specialization
- **6.1 Conceptual Prose (Question 1)**:
  - Evaluation of SVR (MAE 1.3243, QWK 0.5106, $r=0.5767$) and RAG (MAE 1.2568).
  - Capturing semantic completeness without requiring exact verbatim matches.
- **6.2 Mathematical Derivations (Question 2)**:
  - Attention matrix derivations require non-linear step verification.
  - CatBoost (MAE 1.8243) and Stacking (MAE 1.8784) excel; deterministic rules fail on OCR noise.
- **6.3 Algorithmic Execution Traces (Question 3)**:
  - Decoding algorithms evaluated with 86.49% within 1 mark using ElasticNet (MAE 0.6216).
- **6.4 Statistical Laplace Calculations (Question 4)**:
  - Sparse Lasso (MAE 0.8514) eliminates lexical distractions, focusing on probability fractions.
- **Evidence Links**: `tables/model_performance_by_question.csv`, `figures/per_question_performance_heatmap.png`, Claim C01.

### Section VII: Feature Family Ablation Study
- **7.1 Marginal Feature Contributions**:
  - Full feature set (MAE 1.3007, QWK 0.7572) vs. 10 ablation variants.
- **7.2 The Structural Length Anchor**:
  - Dramatic collapse upon removing structural length features ($\Delta \text{QWK} = -0.0654$).
  - Psychological alignment with human grader cognitive anchors.
- **7.3 Symbolic and Semantic Synergy**:
  - Symbolic features prevent mathematical hallucination ($\Delta \text{QWK} = -0.0203$).
  - Full multimodal integration outperforms pure lexical models by $+0.0550$ QWK.
- **Evidence Links**: `tables/feature_ablation_summary.csv`, `figures/feature_family_ablation_waterfall.png`, Claims C03, C04.

### Section VIII: Input Representation Sensitivity & Noise Robustness
- **8.1 Clean Gold vs. TrOCR vs. Tesseract**:
  - TrOCR achieves MAE 1.3007 (QWK 0.7572) vs. Tesseract MAE 1.4426 (QWK 0.7022).
  - Character omission as unintended regularizer vs. character corruption as noise.
- **8.2 Collapse of Rule Graders on Handwritten Text**:
  - Deterministic rule grading drops to QWK $0.0000$ across all OCR formats.
- **8.3 Controlled CER Noise Degradation**:
  - Graceful linear degradation from 0% to 10% CER (MAE 1.1858 $\to$ 1.1959).
- **8.4 Mathematical Operator Inversion Sensitivity**:
  - Lexical blindness to sign flips ($+$ vs. $-$) proves the necessity of symbolic verifiers.
- **Evidence Links**: `tables/modality_comparison_results.csv`, `tables/robustness_perturbation_results.csv`, `figures/modality_sensitivity_slope.png`, `figures/robustness_noise_decay_curves.png`, Claims C05, C06, C07, C08.

### Section IX: Computational Pareto Efficiency & Deployment Economics
- **9.1 Accuracy vs. Latency Trade-Off**:
  - Ridge regression ($0.10\text{ ms}$, MAE 1.3007) and CatBoost ($7.42\text{ ms}$, MAE 1.1926) define the frontier.
- **9.2 Memory Footprint and Hardware Requirements**:
  - $35\text{ MB}$ CPU RAM (Ridge) vs. $3.8\text{ GB}$ GPU VRAM (LLM).
- **9.3 Throughput Comparison**:
  - 10,000 evaluations/second (Ridge) vs. 0.14 evaluations/second (Llama-3.2:3b).
  - Economic feasibility for institutional grading workflows.
- **Evidence Links**: `tables/pareto_efficiency_summary.csv`, `figures/pareto_frontier_accuracy_latency.png`, Claim C09.

### Section X: Discussion, Limitations, & Conclusion
- **10.1 Key Takeaways for Automated Examination Grading Systems**:
  - Deploy hybrid ensembles routed by question type rather than monolithic LLMs.
  - Always couple OCR with learned probabilistic calibrators and symbolic verifiers.
- **10.2 Limitations**:
  - Single-institution cohort (IIIT Dharwad, 37 students, Computer Science domain).
  - Hardware constraint preventing 4-bit NF4 QLoRA fine-tuning on Windows AMD64.
- **10.3 Future Work**:
  - Multi-institutional expansion across broader STEM domains (Physics, Electrical Engineering).
  - Fine-tuning 8B parameter vision-language models directly on cropped page regions.

---

## Mapping of Claims to Manuscript Evidence

| Claim ID | Manuscript Location | Primary Evidence Table | Primary Figure | Status |
| :---: | :---: | :---: | :---: | :---: |
| **C01** | Section V, VI | `tables/model_performance_by_question.csv` | `figures/per_question_performance_heatmap.png` | **VERIFIED_LIVE** |
| **C02** | Section V | `tables/main_model_comparison.csv` | `figures/overall_model_performance_qwk_mae.png` | **VERIFIED_LIVE** |
| **C03** | Section VII | `tables/feature_ablation_summary.csv` | `figures/feature_family_ablation_waterfall.png` | **VERIFIED_LIVE** |
| **C04** | Section VII | `tables/feature_ablation_summary.csv` | `figures/feature_family_ablation_waterfall.png` | **VERIFIED_LIVE** |
| **C05** | Section VIII | `tables/modality_comparison_results.csv` | `figures/modality_sensitivity_slope.png` | **VERIFIED_LIVE** |
| **C06** | Section VIII | `tables/modality_comparison_results.csv` | `figures/modality_sensitivity_slope.png` | **VERIFIED_LIVE** |
| **C07** | Section VIII | `tables/robustness_perturbation_results.csv` | `figures/robustness_noise_decay_curves.png` | **VERIFIED_LIVE** |
| **C08** | Section VIII | `tables/robustness_perturbation_results.csv` | `figures/robustness_noise_decay_curves.png` | **VERIFIED_LIVE** |
| **C09** | Section IX | `tables/pareto_efficiency_summary.csv` | `figures/pareto_frontier_accuracy_latency.png` | **VERIFIED_LIVE** |
| **C10** | Section V, IX | `tables/main_model_comparison.csv` | `figures/overall_model_performance_qwk_mae.png` | **VERIFIED_LIVE** |
