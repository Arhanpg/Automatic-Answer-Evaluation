"""
src/frontier/mamba_vs_transformer.py - Milestone M11: Selective State-Space (Mamba) vs Transformer Benchmark

Hypothesis: Selective State-Space Models (Mamba blocks) scale linearly O(L) in sequence length
and memory compared to Transformer's O(L^2) attention, enabling efficient multi-page exam processing
on a 4GB VRAM GPU with matched representation fidelity.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.output_manager import DedicatedOutputManager


class SelectiveSSMBlock:
    """
    Selective State-Space Model (Mamba block) in NumPy:
    h_t = A_bar * h_{t-1} + B_bar * x_t
    y_t = C_t * h_t + D * x_t
    where B, C, Delta are input-dependent projections.
    """

    def __init__(self, d_model: int = 64, d_state: int = 16, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.d_model = d_model
        self.d_state = d_state

        # Continuous diagonal A: initialized to negative values for stability
        self.A_log = np.log(np.repeat(np.arange(1, d_state + 1)[np.newaxis, :], d_model, axis=0))
        self.A = -np.exp(self.A_log) # (d_model, d_state)

        # Projections for selective B, C, Delta
        self.W_b = rng.randn(d_model, d_state) * 0.05
        self.W_c = rng.randn(d_model, d_state) * 0.05
        self.W_delta = rng.randn(d_model, d_model) * 0.05

        self.D = np.ones(d_model)

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        seq_len, d_model = x_seq.shape
        h = np.zeros((d_model, self.d_state), dtype=np.float32)
        y_seq = np.zeros_like(x_seq)

        # Input-dependent Delta, B, C (Selection Mechanism)
        delta = np.log(1.0 + np.exp(np.dot(x_seq, self.W_delta))) # softplus
        B = np.dot(x_seq, self.W_b) # (seq_len, d_state)
        C = np.dot(x_seq, self.W_c) # (seq_len, d_state)

        # Recurrent scan over sequence length (O(L) linear time)
        for t in range(seq_len):
            dt = delta[t, :, np.newaxis] # (d_model, 1)
            # Discretized A_bar = exp(dt * A)
            A_bar = np.exp(dt * self.A)
            # Discretized B_bar = dt * B_t
            B_bar = dt * B[t, np.newaxis, :]
            # Update state
            h = A_bar * h + B_bar * x_seq[t, :, np.newaxis]
            # Output y_t = C_t h_t + D x_t
            y_seq[t] = np.sum(C[t, np.newaxis, :] * h, axis=-1) + self.D * x_seq[t]

        return y_seq


def benchmark_mamba_vs_transformer() -> List[Dict[str, Any]]:
    """Runs throughput and memory benchmark across multi-page sequence lengths."""
    seq_lengths = [128, 512, 1024, 2048]
    d_model = 64
    results = []

    mamba = SelectiveSSMBlock(d_model=d_model, d_state=16)

    for L in seq_lengths:
        x = np.random.randn(L, d_model).astype(np.float32)

        # 1. Mamba Selective SSM
        t0 = time.perf_counter()
        _ = mamba.forward(x)
        t_mamba_ms = (time.perf_counter() - t0) * 1000.0
        mamba_mem_mb = (L * d_model * 4 + d_model * 16 * 4) / (1024 * 1024)

        # 2. Transformer Attention Simulation: O(L^2) memory and operations
        t0 = time.perf_counter()
        # Simulated Q K^T matrix multiplication: (L, L)
        scores = np.dot(x, x.T) / np.sqrt(d_model)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        _ = np.dot(attn, x)
        t_trans_ms = (time.perf_counter() - t0) * 1000.0
        trans_mem_mb = (L * L * 4 + 2 * L * d_model * 4) / (1024 * 1024)

        results.append({
            "sequence_length": L,
            "mamba_latency_ms": round(t_mamba_ms, 2),
            "transformer_latency_ms": round(t_trans_ms, 2),
            "mamba_memory_mb": round(mamba_mem_mb, 4),
            "transformer_memory_mb": round(trans_mem_mb, 4),
            "mamba_speedup_factor": round(t_trans_ms / max(1e-4, t_mamba_ms), 2),
            "transformer_memory_blowup": round(trans_mem_mb / max(1e-4, mamba_mem_mb), 2)
        })

    return results


if __name__ == "__main__":
    res = benchmark_mamba_vs_transformer()
    for r in res:
        print(r)
