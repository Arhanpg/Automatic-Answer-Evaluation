# Paper 2 Experiment Reproduction Runbook

**Project**: Beyond a Universal Grader: Feature- and Question-Conditional Performance of Models for Handwritten Technical Examination Assessment  
**Repository**: `https://github.com/Arhanpg/Automatic-Answer-Evaluation`  
**Authors**: Arhan Ghosarwade and Abhinav Gumgol (IIIT Dharwad)  
**Git Branch**: `research/paper2-model-comparison`  
**Target Environment**: Windows 11 / AMD64, NVIDIA GPU (RTX 3050 Laptop or better, $\ge 4$ GB VRAM), 16 GB RAM, Python 3.12.x  

---

## 1. Prerequisites and Environment Setup

### 1.1 Python Virtual Environment
Activate the project virtual environment:
```powershell
# From the repository root:
cd "f:\Automatic Answer evaluation\automatic-answer-evaluation\automatic-answer-evaluation"
& "f:\Automatic Answer evaluation\.venv\Scripts\Activate.ps1"
```

Verify installed core packages:
```powershell
python -c "import torch, sklearn, catboost, lightgbm, xgboost, sympy, sentence_transformers, matplotlib; print('Environment dependencies verified successfully!')"
```

### 1.2 Ollama Local Daemon (For Prompted LLM Benchmarking)
Ollama must be installed and running locally with the target model:
```powershell
# Verify Ollama service
ollama list

# Ensure llama3.2:3b is available (or pull if missing)
ollama pull llama3.2:3b

# Pre-warm model in memory
ollama run llama3.2:3b "Hi"
```

---

## 2. Test Suite Execution (Unit & Anti-Leakage Verification)

Before running any benchmarks, execute the 11 pre-registered unit tests to confirm:
- Disjoint student cross-validation splits.
- In-fold feature fit with zero target label leakage.
- Strict exemplar exclusion in RAG candidate pools.
- Exact Robertson BM25Okapi calculation.
- Exact DP-based LCS ROUGE-L calculation.
- SBERT availability and SymPy math verification.
- Discrete QWK and student-clustered bootstrap logic.

```powershell
python -m pytest research_v4/model_comparison/tests/ -v
```
*Expected Result*: `11 passed in ~40 seconds` with zero warnings.

---

## 3. Running the Master Benchmark Suite

To execute the entire 6-experiment benchmark suite:
```powershell
python -m research_v4.model_comparison.src.training.cross_validation
```

### What This Script Executes:
1. **Experiment 1 (Primary 5-Fold Benchmark)**:
   - Evaluates all 22 models across 5 `GroupKFold` splits (grouped strictly by student ID).
   - Evaluates Non-Learning Baselines, Classical ML (Ridge, Lasso, ElasticNet, SVR, RF, ExtraTrees, GBDT, CatBoost, LightGBM, XGBoost), Deep Learning (PyTorch MLP, PyTorch BiLSTM), RAG (BM25, Dense, Hybrid), and Prompted LLM (`llama3.2:3b`).
   - Fits out-of-fold `ValidationTunedStacking` meta-regressor and `ParetoOptimalEnsemble`.
   - Computes $B=2,000$ student-clustered bootstrap 95% CIs and paired cluster permutation tests vs. Ridge.
   - Saves `tables/main_model_comparison.csv`.
2. **Experiment 2 (Question-Conditional Breakdown)**:
   - Evaluates models individually across Q1 (Prose), Q2 (Math), Q3 (Algorithms), and Q4 (Statistics).
   - Saves `tables/model_performance_by_question.csv`.
3. **Experiment 3 (Feature Family Ablations)**:
   - Executes 11 pre-registered feature family masks on Ridge and LightGBM.
   - Saves `tables/feature_ablation_summary.csv`.
4. **Experiment 4 (Input Modality Sensitivity)**:
   - Compares Clean Human Transcripts, TrOCR Hypotheses, and Tesseract Hypotheses.
   - Saves `tables/modality_comparison_results.csv`.
5. **Experiment 5 (Controlled Robustness Stress Tests)**:
   - Evaluates Ridge under synthetic character noise (0%, 1%, 5%, 10% CER) and mathematical sign flips.
   - Saves `tables/robustness_perturbation_results.csv`.
6. **Experiment 6 (Computational Pareto Efficiency)**:
   - Computes latency, memory consumption, parameter counts, and Pareto dominance flags.
   - Saves `tables/pareto_efficiency_summary.csv`.
7. **Telemetry & Predictions**:
   - Persists per-response predictions to `runs/<run_id>/model_predictions.jsonl`.
   - Records hardware and runtime telemetry to `runs/<run_id>/telemetry.json`.

*Expected Execution Time*: ~20–25 minutes on NVIDIA RTX 3050 Laptop GPU.

---

## 4. Generating Publication Figures

To generate all 6 publication figures (300 DPI, publication styling):
```powershell
python -m research_v4.model_comparison.src.reporting.plot_figures
```

Output figures in `research_v4/model_comparison/figures/`:
1. `overall_model_performance_qwk_mae.png`: Overall MAE and QWK with 95% Clustered Bootstrap CIs.
2. `per_question_performance_heatmap.png`: Model $\times$ Question typology performance matrix.
3. `feature_family_ablation_waterfall.png`: Active feature count vs. MAE impact.
4. `modality_sensitivity_slope.png`: Clean Transcript vs. TrOCR vs. Tesseract sensitivity.
5. `robustness_noise_decay_curves.png`: Performance decay under progressive character substitution noise.
6. `pareto_frontier_accuracy_latency.png`: Accuracy vs. Latency Pareto trade-off curve.

---

## 5. Artifact Directory Map

```
research_v4/model_comparison/
├── README.md                                    <- Overview and quickstart
├── audit/                                       <- Provenance and audit documentation
│   ├── annotation_provenance.csv
│   ├── dataset_provenance.csv
│   ├── input_modality_matrix.csv
│   ├── model_reproducibility_matrix.csv
│   ├── privacy_audit.md
│   ├── repository_audit.md
│   └── split_exposure_audit.json
├── protocol/                                    <- Preregistration and claim matrices
│   ├── claim_to_evidence_matrix.csv
│   ├── evidence_policy.md
│   ├── experiment_config.yaml
│   ├── preregistration.md
│   └── research_questions.yaml
├── src/                                         <- Production source code
│   ├── data_loader.py                           <- Leakage-safe data loader
│   ├── splits.py                                <- GroupKFold cross-validation
│   ├── text_normalization.py                    <- Math-preserving tokenizer & true LCS
│   ├── evaluation/metrics.py                    <- Discrete QWK, MAE, RMSE, Pearson r
│   ├── statistics/bootstrap.py                  <- Cluster bootstrap & permutation tests
│   ├── feature_extractors/                      <- BM25, Lexical, Semantic, Symbolic, Structural
│   ├── model_adapters/                          <- Baselines, Classical, Neural, RAG, LLM, Ensembles
│   ├── training/cross_validation.py             <- Master benchmark orchestration runner
│   └── reporting/plot_figures.py                <- 300 DPI publication figure generator
├── tests/                                       <- Unit test suite (11 tests, 100% pass)
├── tables/                                      <- Machine-readable experimental results (CSVs)
├── figures/                                     <- High-resolution publication figures (PNGs)
├── runs/                                        <- Raw execution telemetry and JSONL predictions
└── reports/                                     <- Detailed research reports and manuscript guides
    ├── FINAL_MODEL_COMPARISON_REPORT.md         <- Comprehensive empirical report
    ├── EXPERIMENT_RUNBOOK.md                    <- This execution runbook
    └── PAPER2_OUTLINE.md                        <- Manuscript drafting guide
```
