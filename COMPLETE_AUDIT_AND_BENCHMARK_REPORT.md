# Complete Audit and Benchmark Report

## Scope

This report is the canonical end-to-end audit of the repository. Detailed
milestone reports are stored in [`experiments/reports/`](experiments/reports/).
Raw machine-readable artifacts are in [`outputs/`](outputs/) and
[`experiments/runs/`](experiments/runs/).

## Evidence classification

| Evidence class | Examples | Interpretation |
|---|---|---|
| Recorded implementation output | split manifest, JSON/CSV outputs, gradient verification | Reproduceable repository artifact |
| Scripted/simulated benchmark | independent OCR variations, QLoRA volume ablation, model leaderboard | Useful internal comparison, not live external-model proof |
| Live telemetry | hardware registry, live OCR metrics | Only the recorded run environment and scripts are authoritative |

## Dataset

The corpus contains 37 students and 148 four-question responses. The split is
25 train, 6 validation, and 6 locked test. The split has SHA-256
`41327d9f77e8a40dcc6c027a9250eed79175cc4b6fa2cbdd6f4cc479f40c06e1`.

## Preprocessing and OCR

The implementation order is red-ink separation, grid suppression and deskew,
NLM denoising, and Sauvola binarization. Recorded independent outputs show:

| Engine | CER | WER | Character accuracy |
|---|---:|---:|---:|
| PaddleOCR processed | 1.2% | 5.27% | 98.8% |
| Tesseract processed | 6.6% | 22.14% | 93.4% |
| TrOCR-ViT processed | 0.38% | 0.0% | 99.62% |

Raw live OCR telemetry also exists, but it records substantially worse CER
and WER for s1/s2 and cannot replace a complete independent OCR benchmark.

## Model performance

The primary CSV records:

| Model | MAE | RMSE | Within 1 mark | QWK |
|---|---:|---:|---:|---:|
| Master hybrid ensemble | 0.420 | 0.580 | 95.8% | 0.941 |
| SymPy verifier | 0.480 | 0.650 | 93.5% | 0.928 |
| Hybrid RAG | 0.550 | 0.760 | 93.0% | 0.915 |
| Transformer from scratch | 0.469 | 0.685 | 93.8% | 0.912 |
| QLoRA 25 + synthetic | 0.820 | 1.120 | 91.7% | 0.906 |
| BiLSTM with attention | 0.477 | 0.720 | 92.5% | 0.898 |
| Ridge | 1.473 | 1.724 | 84.5% | 0.7591 |

The highest overall generated score is the Master Hybrid Ensemble.
Symbolic verification and constrained math handling explain much of the value
of the hybrid approach. The leaderboard is generated from the repository
benchmark, not a claim that every named model was trained on hardware.

## QLoRA data effect

| Training scripts | MAE | RMSE | QWK | Within 1 mark |
|---:|---:|---:|---:|---:|
| 5 | 1.62 | 2.14 | 0.742 | 62.5% |
| 15 | 1.18 | 1.58 | 0.835 | 79.2% |
| 25 | 0.94 | 1.27 | 0.884 | 87.5% |
| 25 + synthetic | 0.82 | 1.12 | 0.906 | 91.7% |

The hardware registry does not show live GPU training; it records CPU
execution and 0 MB peak VRAM.

## Reliability limitations

The conformal artifact contains 24 test predictions and 1-mark intervals, but
records zero calibration samples. It therefore does not establish a valid
calibrated 90% coverage guarantee. Any deployment study must use a real
calibration partition and retained test partition.

## Recommended final claims

It is accurate to state that the repository implements:

- end-to-end parsing, OCR, red-ink separation, grading, uncertainty output,
  Flask dashboards, and downloadable benchmark artifacts;
- student-level leakage protection;
- explicit rubric and symbolic-math verification;
- a complete 27-model comparison artifact;
- gradient and split validation.

It is not accurate to claim that every external OCR and GPU model value was
measured live. It is also not accurate to present the 100% RAG or conformity
numbers as universal real-world guarantees.
