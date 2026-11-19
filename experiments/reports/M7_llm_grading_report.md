# Milestone M7: Evidence-Constrained LLM Grading Report

## Benchmark evidence

The LLM benchmark summary reports 12 response instances for the prompt/RAG
experiments. Results include:

| Experiment | MAE | RMSE | QWK | Status |
|---|---:|---:|---:|---|
| Zero-shot reconciled rubric | 0.250 | 0.500 | 0.9746 | Recorded |
| Few-shot, 3 examples | 0.230 | 0.470 | 0.9946 | Recorded; clean pool asserted |
| Criterion-decomposed | 0.225 | 0.455 | 1.0000 | Recorded |
| Two-turn self-critique | 0.220 | 0.445 | 1.0000 | Recorded |
| Multi-sample ensemble | 0.2125 | 0.430 | 1.0000 | Recorded |

These are benchmark-artifact values, not a 24-example locked-test re-run from
the actual Llama or Ollama runtime.

## Targeted checks

The generated summary records:

- s2 and s29 Q4 math answers receive 3/3.
- s29 Q3(b) contradiction receives 0/1.
- A prompt-injection test changes no score.
- The s2 Q3(c) alternative-math flag is recorded as false in the raw JSON,
  so it must not be stated that the experiment passed this individual check.

## Interpretation

Criterion decomposition and ensemble routing reduce MAE in the recorded
ablation. However, the evidence is not enough to claim that an independently
deployed Ollama model is guaranteed to reproduce it. Deployment depends on
the selected model, prompt version, JSON-enforcement behavior, and actual OCR
input quality.
