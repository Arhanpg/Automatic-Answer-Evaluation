# Phase 14: Reproducibility Protocol & Experimental Defense Bundle

## Automatic Answer Evaluation: Assessment-Critical Perception, Routing, and Risk Control

**Authors**: Arhan Ghosarwade and Abhinav Gumgol  
**Affiliation**: Department of Computer Science & Engineering, Indian Institute of Information Technology Dharwad (IIIT Dharwad), Karnataka, India  
**Target Venue**: *IEEE Transactions on Artificial Intelligence* (IEEE TAI)  
**Artifact Directory**: `manuscript_v3/`  
**Repository Branch**: `research/phase13-manuscript-synthesis`

---

## 1. System & Hardware Specifications

The empirical benchmarks reported in this investigation were executed and validated on the following physical environment:

- **Operating System**: Microsoft Windows 11 Enterprise / Home (Build 26100)
- **Host CPU**: AMD Ryzen 7 / Intel Core i7 Multi-Core Processor
- **Host GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM, CUDA Compute Capability 8.6)
- **CUDA Toolchain**: CUDA 12.4 with PyTorch `2.6.0+cu124`
- **Python Runtime**: Python 3.12.10 (64-bit) in virtual environment `.venv`
- **OCR Engines**:
  - `microsoft/trocr-small-printed` / `microsoft/trocr-small-handwritten` (Hugging Face Transformers 4.49.0)
  - Tesseract OCR v5.5.0.20241111 (libtesseract 5.5.0, Leptonica 1.85.0)
  - PaddleOCR v2.9.1 (PaddlePaddle-GPU 3.0.0b2; operational limitation documented on Windows PIR oneDNN runtime)
- **Grading & NLP Libraries**: Sentence-Transformers 3.4.1 (`all-MiniLM-L6-v2`), SymPy 1.13.3, Scikit-Learn 1.6.1, OpenCV 4.11.0, SciPy 1.15.2

---

## 2. Dataset Architecture & Provenance

The primary corpus consists of authentic, unconstrained handwritten university examination booklets administered at IIIT Dharwad for an advanced Computer Science course on *Large Language Models*:

| Attribute | Value | Description |
|:---|:---|:---|
| **Total Booklets** | 37 | Complete multi-page examination scripts (`s1` through `s37`) |
| **Total Scanned Pages** | 195 | High-resolution RGB scans ($3072 \times 4096$ or $4096 \times 3072$) |
| **Total Subquestion Slots** | 407 | 11 subquestion items per student across 4 parent questions |
| **Written Subquestions** | 392 | Authentic student handwriting responses |
| **Omitted Subquestions** | 15 | Explicit non-responses (verified via manual forensic review) |
| **Parent Grading Slots** | 148 | 4 parent questions per student evaluated out of 20.0 total marks |
| **Institutional Split** | 25 / 6 / 6 | 25 Train (67.6%), 6 Validation (16.2%), 6 Sequestered Locked Test (16.2%) |
| **Locked Test IDs** | `s7`, `s14`, `s24`, `s29`, `s30`, `s37` | Strict zero-tuning sequestered evaluation set |

### Privacy & Governance Compliance
Under strict privacy regulations and institutional consent, raw scans and identified personal student data are quarantined in local private storage (`F:\Automatic answer evaluation dataset`). All public evaluation artifacts, CSV matrices, and manuscript tables use anonymized student tokens `s1` to `s37`.

---

## 3. End-to-End Reproduction Pipeline

To reproduce all tables, figures, statistical tests, and artifacts from scratch, execute the phase scripts in sequential order:

```powershell
# Activate Python 3.12 virtual environment
& "F:\Automatic Answer evaluation\.venv\Scripts\Activate.ps1"
Set-Location "F:\Automatic Answer evaluation\automatic-answer-evaluation\automatic-answer-evaluation"

# Set Dataset Root environment variable
$env:AAE_DATA_ROOT = "F:\Automatic answer evaluation dataset"
```

### Stage 1: Phase 8 & 8.1 Ground-Truth Reconstruction
Validates dual-rater independent blind trials, inter-annotator agreement, and freezes consensus ground-truth annotations:
```powershell
python research_v3/phase8_ground_truth/scripts/01_preflight_audit.py
python research_v3/phase8_ground_truth/scripts/02_dual_rater_adjudication.py
python research_v3/phase8_ground_truth/scripts/03_calculate_agreement.py
python research_v3/phase8_ground_truth/scripts/04_split_integrity.py
```
*Outputs*:
- `research_v3/phase8_ground_truth/03_INTER_ANNOTATOR_AGREEMENT.json`
- `research_v3/phase8_ground_truth/04_DATASET_FREEZE_REPORT.md`
- `research_v3/phase8_ground_truth/06_SPLIT_INTEGRITY_REPORT.md`

### Stage 2: Phase 9 Multi-Model Isolated OCR Benchmarking
Executes 2,442 live inference passes across TrOCR-Small (CUDA), Tesseract, and PaddleOCR across Raw and Preprocessed conditions:
```powershell
python research_v3/phase9_ocr_benchmarking/scripts/run_ocr_benchmark.py
python research_v3/phase9_ocr_benchmarking/scripts/compute_ocr_metrics.py
python research_v3/phase9_ocr_benchmarking/scripts/run_statistical_tests.py
```
*Outputs*:
- `research_v3/phase9_ocr_benchmarking/results/ocr_metrics_by_engine.csv`
- `research_v3/phase9_ocr_benchmarking/results/paired_statistical_tests.csv`

### Stage 3: Phase 10 Answer-Region Routing & Rotation Invariance
Evaluates layout segmentation across upright and sideways-rotated scans ($270^\circ / 180^\circ$ in `s7`, `s16`, `s34`, `s37`):
```powershell
python research_v3/phase10_routing/scripts/evaluate_routing.py
```
*Outputs*:
- `research_v3/phase10_routing/results/routing_metrics_by_system.csv`
- `research_v3/phase10_routing/results/rotation_ablation.csv`
- `research_v3/phase10_routing/results/iou_distribution.csv`

### Stage 4: Phase 11 Controlled 2x2 Factorial Error Propagation
Evaluates Streams A, B, C, D-Baseline, and D-Hybrid under frozen rubric scoring; executes two-way ANOVA and Shapley decomposition:
```powershell
python research_v3/phase11_error_propagation/scripts/run_factorial_propagation.py
```
*Outputs*:
- `research_v3/phase11_error_propagation/results/stagewise_summary.csv`
- `research_v3/phase11_error_propagation/results/anova_decomposition_summary.csv`
- `research_v3/phase11_error_propagation/results/question_level_breakdown.csv`

### Stage 5: Phase 12 Input-Side Calibrated Risk Control
Calculates composite uncertainty without gold mark leakage; computes selective review curves across 12 review budgets:
```powershell
python research_v3/phase12_risk_control/scripts/run_risk_control_pipeline.py
```
*Outputs*:
- `research_v3/phase12_risk_control/results/calibration_correlation.csv`
- `research_v3/phase12_risk_control/results/subgroup_risk_profile.csv`
- `research_v3/phase12_risk_control/results/selective_review_curves.csv`

---

## 4. Automated Defense & Verification Test Suite

To verify that all 14 empirical claims in the manuscript match the verified CSV values on disk with zero discrepancies:

```powershell
python -m pytest manuscript_v3/tests/test_manuscript_defense.py -v
```

To run the complete system-wide test suite across all research phases:
```powershell
python -m pytest research_v3/phase8_ground_truth/tests/ research_v3/phase9_ocr_benchmarking/tests/ research_v3/phase10_routing/tests/ research_v3/phase11_error_propagation/tests/ research_v3/phase12_risk_control/tests/ manuscript_v3/tests/ -v
```

---

## 5. Summary of Core Empirical Findings

1. **The Preprocessing Paradox**: Classical binarization improves heuristic OCR (Tesseract CER $2.5042 \rightarrow 1.0000, p < 10^{-5}$) but significantly impairs modern Vision Transformers (TrOCR CER $1.1687 \rightarrow 1.3762, p = 0.00928$), demonstrating that deep patch encoders rely on raw anti-aliased edge gradients.
2. **Scan Orientation Fragility**: Sideways scanning causes naive regex layout parsers to fail catastrophically ($\text{IoU} = 0.0793$, 0.0% coverage), falsely treating written exams as omitted. The proposed hybrid continuation parser completely recovers $100\%$ slot accuracy and eliminates cross-question contamination.
3. **The Dominance of Recognition Error**: Factorial ANOVA reveals that perceptual recognition accounts for 82.05% of grading variance ($F = 658.03, p = 1.46 \times 10^{-55}$) and over $100\%$ of pipeline mark destruction under Shapley attribution ($\phi_{\text{rec}} = 104.4\%$).
4. **Prose vs Mathematical Asymmetry**: Recognition noise destroys conceptual prose ($Q1$ MAE $+5.0270$m, accounting for 82.3% of OCR loss) while leaving mathematical derivations invariant ($Q2$ MAE unchanged).
5. **Calibrated Selective Human Review**: Escalating the top 20% most ambiguous responses to human examiners reduces human grading workload by 81.1% while slashing total error to 10.0676 marks; a 75% review budget achieves substantial grading agreement ($\text{QWK} = 0.6436, \text{MAE} = 2.3378$).
