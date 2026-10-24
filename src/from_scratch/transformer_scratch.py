"""
src/from_scratch/transformer_scratch.py - Milestone M5: Transformer Components from Scratch in NumPy

Mirroring the IIIT Dharwad Examination Concepts:
1. Scaled Dot-Product Self-Attention: Softmax((Q K^T)/sqrt(d_k)) V (Question 2)
2. Sinusoidal Positional Encoding: sin/cos wave formulas across dimension i (Question 1b)
3. Causal Attention Masking: Upper-triangular future token masking (Question 1d)
4. Dual Layer Normalization: LayerNorm applied twice per block for Attention & FFN (Question 3c)
5. Multi-Head Attention (MHA) and Feed-Forward Networks (FFN)
"""

import numpy as np
from typing import Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ==========================================
# 1. Sinusoidal Positional Encoding (Q1b)
# ==========================================

class SinusoidalPositionalEncoding:
    """
    Computes deterministic sinusoidal positional embeddings (Vaswani et al., 2017):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512):
        self.d_model = d_model
        self.max_len = max_len

        pe = np.zeros((max_len, d_model), dtype=np.float64)
        position = np.arange(0, max_len, dtype=np.float64)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, seq_len: int) -> np.ndarray:
        return self.pe[:seq_len]


# ==========================================
# 2. Causal Mask Generator (Q1d)
# ==========================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Constructs an upper-triangular causal mask matrix M of shape (seq_len, seq_len).
    Zeros on and below diagonal; -1e9 strictly above diagonal to prevent future token leakage.
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return np.where(mask == 1, -1e9, 0.0)


# ==========================================
# 3. Scaled Dot-Product Self-Attention (Q2)
# ==========================================

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = Softmax((Q K^T) / sqrt(d_k) + M) V

    Q: (..., N, d_k)
    K: (..., M, d_k)
    V: (..., M, d_v)
    Returns: (output: (..., N, d_v), attention_weights: (..., N, M))
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V)
    return output, weights


# ==========================================
# 4. Layer Normalization (Q3c)
# ==========================================

class LayerNorm:
    """
    Applies Layer Normalization across the feature dimension:
    LN(x) = ((x - mu) / sqrt(sigma^2 + eps)) * gamma + beta
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


# ==========================================
# 5. Multi-Head Attention (MHA)
# ==========================================

class MultiHeadAttention:
    """Multi-Head Attention supporting causal and bidirectional attention."""

    def __init__(self, d_model: int, num_heads: int = 4, seed: int = 42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / d_model)

        self.W_q = rng.randn(d_model, d_model) * scale
        self.W_k = rng.randn(d_model, d_model) * scale
        self.W_v = rng.randn(d_model, d_model) * scale
        self.W_o = rng.randn(d_model, d_model) * scale

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Q, K, V: (batch_size, seq_len, d_model)
        batch_size, seq_len_q, _ = Q.shape
        _, seq_len_k, _ = K.shape

        q_proj = np.dot(Q, self.W_q).reshape(batch_size, seq_len_q, self.num_heads, self.d_k).swapaxes(1, 2)
        k_proj = np.dot(K, self.W_k).reshape(batch_size, seq_len_k, self.num_heads, self.d_k).swapaxes(1, 2)
        v_proj = np.dot(V, self.W_v).reshape(batch_size, seq_len_k, self.num_heads, self.d_k).swapaxes(1, 2)

        attn_out, attn_weights = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask=mask)

        # Concatenate heads: (batch_size, seq_len_q, d_model)
        concat = attn_out.swapaxes(1, 2).reshape(batch_size, seq_len_q, self.d_model)
        out = np.dot(concat, self.W_o)
        return out, attn_weights


# ==========================================
# 6. Transformer Encoder Block with Dual LN (Q3c)
# ==========================================

class TransformerEncoderBlock:
    """
    Standard Transformer Block featuring Dual Layer Normalization:
    - LayerNorm 1: Normalizes before/after Multi-Head Attention
    - LayerNorm 2: Normalizes before/after Feed-Forward Network
    Stabilizes activations and gradient flow across variable sequence lengths.
    """

    def __init__(self, d_model: int, num_heads: int = 4, d_ff: int = 128, seed: int = 42):
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model, num_heads=num_heads, seed=seed)
        self.ln1 = LayerNorm(d_model) # First LayerNorm (for Attention)
        self.ln2 = LayerNorm(d_model) # Second LayerNorm (for FFN)

        rng = np.random.RandomState(seed + 10)
        # Position-wise Feed-Forward Network
        self.W_ff1 = rng.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b_ff1 = np.zeros(d_ff)
        self.W_ff2 = rng.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b_ff2 = np.zeros(d_model)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Pre-LN sub-layer 1: Multi-Head Attention + Residual Connection
        norm1 = self.ln1.forward(x)
        attn_out, _ = self.mha.forward(norm1, norm1, norm1, mask=mask)
        x = x + attn_out

        # Pre-LN sub-layer 2: Feed-Forward Network + Residual Connection
        norm2 = self.ln2.forward(x)
        ffn_inner = np.maximum(0.0, np.dot(norm2, self.W_ff1) + self.b_ff1) # ReLU
        ffn_out = np.dot(ffn_inner, self.W_ff2) + self.b_ff2
        x = x + ffn_out

        return x
