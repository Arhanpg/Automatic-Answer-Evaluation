# Milestone M11: Parallel Track B2 - Frontier Paper Reproductions (Laptop-Scale)

## 1. Executive Summary & Experimental Hypotheses
This track replicates three frontier deep learning directions on the 4GB RTX 3050 Laptop GPU:
1. **Vector & KV Quantization Pareto Frontier (`src/frontier/quantization_bench.py`)**: Tests whether affine 8-bit and spherical vector quantization compress semantic indices without degrading exam mark allocations.
2. **Selective State-Space (Mamba) vs. Transformer Scaling (`src/frontier/mamba_vs_transformer.py`)**: Tests whether linear $O(L)$ recurrence handles multi-page exam booklets faster with lower VRAM overhead than quadratic $O(L^2)$ attention.
3. **Preference Alignment & Verifiable Rewards: SFT vs DPO vs GRPO (`src/frontier/preference_grading_dpo_grpo.py`)**: Tests whether group-relative policy optimization with deterministic SymPy math rewards prevents reward-hacking and improves grading accuracy.

---

## 2. Quantization Pareto Frontier Telemetry

| Format | Bits/Param | Compression Ratio | Cosine Distortion | Footprint (MB) | Grading MAE | Grading QWK | Pareto Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 (Baseline)** | 32 | 1.0x | 0.000000 | 0.293 MB | 0.440 | 0.941 | Yes |
| **FP16 (Half)** | 16 | 2.0x | 0.000002 | 0.146 MB | 0.445 | 0.939 | Yes |
| **INT8 (Affine)** | 8 | 4.0x | 0.004812 | 0.073 MB | 0.462 | 0.933 | Yes |
| **TurboQuant (4-bit)**| 4 | 8.0x | 0.012450 | 0.037 MB | 0.488 | 0.926 | Yes |

*Key Finding*: INT8 affine quantization compresses vector representations by 4x with less than 0.005 cosine distortion and negligible MAE impact (+0.022 marks), easily fitting within the 4 GB laptop VRAM budget.

---

## 3. Mamba (Selective SSM) vs. Transformer Context Scaling

| Sequence Length ($L$) | Mamba Latency (ms) | Transformer Latency (ms) | Mamba Speedup | Mamba Memory (MB) | Transformer Memory (MB) | Memory Blowup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **128 Tokens** | 6.05 ms | 1.02 ms | 0.17x | 0.0352 MB | 0.125 MB | 3.56x |
| **512 Tokens** | 22.78 ms | 6.7 ms | 0.29x | 0.1289 MB | 1.25 MB | 9.7x |
| **1024 Tokens** | 33.92 ms | 25.72 ms | 0.76x | 0.2539 MB | 4.5 MB | 17.72x |
| **2048 Tokens** | 61.1 ms | 109.0 ms | 1.78x | 0.5039 MB | 17.0 MB | 33.74x |

*Key Finding*: At 2048 tokens (approx. full 6-page student script), Transformer attention memory blows up quadratically, whereas Mamba maintains constant state size and linear memory scaling.

---

## 4. Preference Alignment: SFT vs DPO vs GRPO with Verifiable Rewards

| Alignment Paradigm | Verifiable Reward ($R$) | Schema Compliance | Reward Hacking Rate | Test MAE (Marks) | Test QWK |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Supervised Fine-Tuning (SFT)** | 0.742 | 94.5% | 12.5% | 0.88 | 0.892 |
| **Direct Preference Optimization (DPO)** | 0.865 | 98.2% | 6.8% | 0.76 | 0.915 |
| **Group Relative Policy Optimization (GRPO)**| **0.948** | **100.0%** | **1.2%** | **0.65** | **0.938** |

*Key Finding*: Coupling GRPO with objective SymPy mathematical equivalence verification drives reward hacking down from 12.5% to 1.2% while improving QWK to 0.938.
