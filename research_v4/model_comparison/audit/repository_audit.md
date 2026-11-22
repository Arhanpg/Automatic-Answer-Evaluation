# Source-Level Repository Audit: Model Training, Feature Engineering, and Evaluation Infrastructure

**Auditors**: Arhan Ghosarwade & Abhinav Gumgol  
**Affiliation**: Department of Computer Science & Engineering, IIIT Dharwad  
**Date**: November 22, 2026  
**Audited Baseline**: Commit `cfcdfa1` on branch `research/paper2-model-comparison`

---

## 1. Scope & Purpose

This forensic source-level audit evaluates the machine learning, feature extraction, neural network, prompted LLM, and evaluation code across the `Automatic-Answer-Evaluation` repository. The objective is to identify all methodological flaws, target leakages, simulated metrics, and architectural proxies in legacy code before implementing the clean, reproducible Paper 2 experimental suite.

---

## 2. Exhaustive Audit of Known Problems

### A. Target Leakage in Legacy 27-Model Comparison (`backend/independent_pipeline.py`)

- **File**: `backend/independent_pipeline.py`
- **Function**: `train_and_eval_27_fresh_models(X_train, y_train, X_test, y_test, meta_test, ocr_noise_scale)` (Lines 490–575)
- **Defect Description**:
  While classical estimators (Ridge, ElasticNet, SVR, Random Forest, GBDT, etc.) were genuinely fitted on training folds using scikit-learn, several deep-learning, prompted-LLM, RAG, and fine-tuned models were constructed via linear combinations of the ground-truth test labels (`y_test`) and a GBDT anchor with injected Gaussian noise.
- **Specific Lines Audited**:
  - Line 500: `trans_p = np.clip(0.75 * gbdt_anchor + 0.25 * y_test + np.random.normal(0, 0.14 + ocr_noise_scale * 0.16, size=len(y_test)), 0.0, test_max_marks)` (`Transformer_Encoder`)
  - Line 504: `lstm_p = np.clip(0.65 * gbdt_anchor + 0.35 * y_test + np.random.normal(0, 0.18 + ocr_noise_scale * 0.20, size=len(y_test)), 0.0, test_max_marks)` (`LSTM_Scratch`)
  - Line 508: `cnn_p = np.clip(0.62 * gbdt_anchor + 0.38 * y_test + np.random.normal(0, 0.20 + ocr_noise_scale * 0.22, size=len(y_test)), 0.0, test_max_marks)` (`TextCNN_1D`)
  - Line 512: `rnn_p = np.clip(0.55 * gbdt_anchor + 0.45 * y_test + np.random.normal(0, 0.25 + ocr_noise_scale * 0.25, size=len(y_test)), 0.0, test_max_marks)` (`Vanilla_RNN`)
  - Line 533: `fewshot_p = np.clip(0.70 * gbdt_anchor + 0.30 * y_test + np.random.normal(0, 0.22 + ocr_noise_scale * 0.20, size=len(y_test)), 0.0, test_max_marks)` (`FewShot_CoT_LLM`)
  - Line 534: `llama_p = np.clip(0.68 * gbdt_anchor + 0.32 * y_test + np.random.normal(0, 0.24 + ocr_noise_scale * 0.22, size=len(y_test)), 0.0, test_max_marks)` (`Llama-3-8B`)
  - Line 535: `mistral_p = np.clip(0.65 * gbdt_anchor + 0.35 * y_test + np.random.normal(0, 0.26 + ocr_noise_scale * 0.24, size=len(y_test)), 0.0, test_max_marks)` (`Mistral-7B`)
  - Lines 537–540: `qlora_25_syn_p`, `qlora_25_p`, `qlora_15_p`, `qlora_5_p` all blended $22\%$ to $55\%$ of `y_test`.
- **Verdict**: **CRITICAL TARGET LEAKAGE**. These values are not model predictions; they leak the evaluation target. All such formulas are permanently purged in `research_v4/`.

---

### B. OCR Text Input Substitution in Independent Benchmark (`backend/independent_pipeline.py`)

- **File**: `backend/independent_pipeline.py`
- **Function**: `run_independent_benchmark_suite()` (Lines 695–750)
- **Defect Description**:
  The benchmark was declared as testing grading models on PaddleOCR, Tesseract, and Vision Transformer inputs. However:
  1. For PaddleOCR, text was not extracted from images across the full test set. Lines 700–708 set `ref_ans = " ".join(gold_data[sid]["answers"].values())` and simulated OCR noise by dropping characters every 8 words on only 2 scripts (`s6` and `s18`).
  2. Features were constructed by calling `build_student_features(gold_data, exam_schema)` (Line 740), which reads clean reference transcripts (`gold_data`), not OCR engine outputs.
  3. The OCR CER was used solely as a noise scaling factor (`ocr_noise_scale`) passed into `train_and_eval_27_fresh_models`.
- **Verdict**: **SIMULATED MODALITY PIPELINE**. Paper 2 discards this simulated pathway entirely. Instead, Paper 2 directly consumes the authentic per-answer hypothesis strings from the verified Phase 9 benchmark artifact (`research_v3/phase9_ocr_benchmarking/results/predictions.jsonl`).

---

### C. M7 Prompted-LLM Evaluation Simulation (`backend/grader.py` & `experiments/`)

- **File**: `backend/grader.py`, `backend/ollama_client.py`
- **Defect Description**:
  `EvidenceConstrainedLLMGrader` defined prompt structures, but batch benchmarks in historical reports generated metrics by scaling zero-shot performance rather than running independent API calls per prompt condition.
- **Verdict**: **INCOMPLETE LIVE INFERENCE**. In Paper 2, all LLM evaluations are executed through live local inference against verified local Ollama models (`llama3.2:3b`, `qwen2.5:7b`), with raw prompt strings, raw JSON outputs, latency logs, and parse failure rates recorded.

---

### D. Hardcoded Simulation in M9 QLoRA Trainer (`backend/qlora_trainer.py`)

- **File**: `backend/qlora_trainer.py`
- **Function**: `run_qlora_data_ablation()` (Lines 150–200)
- **Defect Description**:
  Lines 163–174 simulated training loss curves with an exponential decay formula (`max(0.28, start_loss * math.exp(-0.75 * ep) + np.random.normal(0, 0.015))`).
  Lines 178–198 hardcoded test set metrics:
  - 5 scripts: MAE = 1.62, RMSE = 2.14, QWK = 0.742
  - 15 scripts: MAE = 1.18, RMSE = 1.58, QWK = 0.835
  - 25 scripts: MAE = 0.94, RMSE = 1.27, QWK = 0.884
  - 25 + synthetic: MAE = 0.82, RMSE = 1.12, QWK = 0.906
- **Verdict**: **PURE SYNTHETIC SIMULATION**. In Paper 2, QLoRA is audited against hardware realities (Windows lacks `bitsandbytes` support natively, preventing 4-bit NF4 quantized training on this specific device without WSL/Linux). This platform limitation is documented honestly as `NOT_RUN (UNAVAILABLE: bitsandbytes on Windows AMD64)`.

---

### E. Methodological Flaws in Feature Engineering (`backend/feature_engineering.py`)

1. **BM25 Proxy Defect**:
   - Line 60: `bm25_score = float(lex["token_jaccard"] * (1.0 + math.log(max(1, len(s_clean.split())))))`
   - *Audit Finding*: This formula is a heuristic product of Jaccard similarity and logarithmic length. It lacks document frequency, inverted index, term frequency saturation ($k_1$), and document length normalization ($b$). It is not BM25.
   - *Correction*: Replaced with standard Robertson et al. `BM25Okapi` fitted strictly on training folds.
2. **ROUGE-L LCS Estimation Defect**:
   - Lines 30–32: `matcher = rapidfuzz.distance.Levenshtein.distance(s_words, m_words); lcs = max(0, (len(s_words) + len(m_words) - matcher) // 2)`
   - *Audit Finding*: Levenshtein distance allows substitutions at cost 1, whereas Longest Common Subsequence (LCS) only permits insertions and deletions. The formula $(|S| + |M| - \text{Lev}) / 2$ is mathematically incorrect for strings with substitutions.
   - *Correction*: Replaced with dynamic programming / exact LCS calculation.
3. **Dense Semantic Scorer Silent Fallback**:
   - `backend/semantic_scorer.py` (Line 117): When `SentenceTransformer` fails to load, `compute_similarity` silently falls back to `tfidf_cosine_similarity`.
   - *Audit Finding*: The caller is never informed that the embedding feature has collapsed to TF-IDF.
   - *Correction*: Paper 2 verifies model loading explicitly; if a model fails, it raises an exception or logs the failure explicitly with a separate fallback flag.
4. **SymPy Concatenation Passing Flaw**:
   - Line 81: `m_res, _ = math_verifier.verify_q2({"2a": s_clean, "2b": s_clean, "2c": s_clean})`
   - *Audit Finding*: The entire question response `s_clean` was simultaneously passed to subquestions 2a, 2b, and 2c, causing spurious cross-subquestion matching.
   - *Correction*: Subquestions are preserved as distinct inputs when verifying mathematical derivations.

---

### F. Repository Search for User-Mentioned "Hexagonal Feature Extraction"

- **Command**: `git grep -i "hexagonal"` and `git grep -i "hexagon"`
- **Result**: `Exit code 1` (zero matches across all branches and tracked commits).
- **Finding**: No feature named "hexagonal feature extraction" exists in the repository. We classify it as `NOT_FOUND_IN_REPOSITORY`. We will not fabricate or approximate a definition.

---

## 3. Summary of Discarded vs. Approved Methodologies

| Component | Historical Legacy Code | Paper 2 Audited Methodology |
|:---|:---|:---|
| **Deep Learning Predictions** | Injected $y_{\text{test}}$ with Gaussian noise | Live PyTorch models (MLP, BiLSTM) trained with real cross-entropy/MSE loops |
| **Prompted LLMs** | Scaled benchmark projections | Live local Ollama HTTP inference (`llama3.2:3b`, `qwen2.5:7b`) with JSON extraction |
| **BM25 Metric** | Token Jaccard $\times$ Log-Length | Robertson et al. standard `BM25Okapi` fit on training fold corpus |
| **ROUGE-L Metric** | Approximate Levenshtein formula | Exact Longest Common Subsequence dynamic programming |
| **OCR Text Input** | Reference text with artificial character drops | Genuine Phase 9 TrOCR-Small live inference transcripts |
| **Splitting Strategy** | Historical exposed split (`s7, s14, s24, s29, s30, s37`) | 5-fold GroupKFold by student ID (37 students, 148 questions) |
| **Statistical Inference** | Independent row-level ANOVA | Student-clustered bootstrap ($B = 2,000$) and paired permutation tests |
