# Milestone M11: Frontier Reproduction Laboratory

## Scope

M11 contains reproducibility-oriented implementations for quantization,
state-space models, and preference alignment. Values in this report are
artifact outputs and should not be cited as externally reproduced papers.

## Quantization artifact

The quantization table reports:

| Format | Compression | Cosine distortion | Grading MAE | QWK |
|---|---:|---:|---:|---:|
| FP32 | 1x | 0.000000 | 0.440 | 0.941 |
| FP16 | 2x | 0.000002 | 0.445 | 0.939 |
| INT8 | 4x | 0.004812 | 0.462 | 0.933 |
| 4-bit | 8x | 0.012450 | 0.488 | 0.926 |

## Recorded Mamba result

The frontier run JSON contains a 2048-token speedup of 1.43. The broader
hardware registry contains conflicting older speedup values. The `1.43x`
value from the latest named M11 JSON is the internally consistent default
for this report.

## Preference alignment artifact

| Method | Reward | Schema compliance | Reward hacking | MAE | QWK |
|---|---:|---:|---:|---:|---:|
| SFT | 0.742 | 94.5% | 12.5% | 0.88 | 0.892 |
| DPO | 0.865 | 98.2% | 6.8% | 0.76 | 0.915 |
| GRPO | 0.948 | 100.0% | 1.2% | 0.65 | 0.938 |

The values are generated experiment outputs.
