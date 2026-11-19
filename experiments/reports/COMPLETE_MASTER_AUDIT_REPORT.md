# Complete Master Audit Report

This file is the index and evidence caveat for the milestone documentation.
Use [`../../COMPLETE_AUDIT_AND_BENCHMARK_REPORT.md`](../../COMPLETE_AUDIT_AND_BENCHMARK_REPORT.md)
as the master corrected report and use each M1-M11 report for milestone
details.

## Milestone index

| Milestone | Subject | Report |
|---|---|---|
| M1 | Dataset, split, and anti-leakage | [M1](M1_dataset_and_rubric_audit.md) |
| M2 | OCR and preprocessing | [M2](M2_ocr_report.md) |
| M3/M6 | Semantic, NLI, and SymPy baseline | [M3/M6](M3_M6_unified_baseline.md) |
| M4 | Classical ML | [M4](M4_classical_ml_report.md) |
| M5 | From-scratch deep learning | [M5](M5_deep_learning_scratch.md) |
| M7 | LLM grading | [M7](M7_llm_grading_report.md) |
| M8 | RAG | [M8](M8_rag_retrieval_report.md) |
| M9 | QLoRA | [M9](M9_finetuning_report.md) |
| M10 | Reliability | [M10](M10_reliability_report.md) |
| M11 | Frontier reproductions | [M11](M11_frontier_report.md) |

## Provenance rule

Tables with generated, simulated, scripted, or CPU-fallback values must remain
labelled. Live measurements require an actual engine invocation and retained
raw output. This rule prevents the README and reports from overstating the
environment's recorded evidence.
