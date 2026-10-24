"""
src/from_scratch/cnn_rnn_lstm_scratch.py - Milestone M5: Text CNN, Vanilla RNN, LSTM, GRU & BiLSTM Attention

Features:
- 1D Text Convolution with Max-Pooling over time
- Step-by-step unrolled Vanilla RNN with BPTT
- Complete LSTM with all 4 gates (input, forget, candidate, output) and cell state updates
- Gated Recurrent Unit (GRU) with reset and update gates
- Bidirectional LSTM (BiLSTM) with additive sequence attention
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ==========================================
# 1D Text Convolutional Network
# ==========================================

class TextCNN1D:
    """1D Convolution over word embedding sequences with temporal max-pooling."""

    def __init__(self, embed_dim: int, num_filters: int = 16, kernel_size: int = 3, out_dim: int = 1, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.embed_dim = embed_dim

        # Weights: (num_filters, kernel_size, embed_dim)
        limit = np.sqrt(2.0 / (kernel_size * embed_dim))
        self.W = rng.randn(num_filters, kernel_size, embed_dim) * limit
        self.b = np.zeros(num_filters)

        # Dense classification head
        self.W_proj = rng.randn(num_filters, out_dim) * np.sqrt(2.0 / num_filters)
        self.b_proj = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (seq_len, embed_dim)
        Returns: (out_dim,)
        """
        seq_len, _ = x.shape
        if seq_len < self.kernel_size:
            # Pad sequence if shorter than kernel
            pad_width = ((0, self.kernel_size - seq_len), (0, 0))
            x = np.pad(x, pad_width, mode="constant")
            seq_len = self.kernel_size

        feature_maps = []
        for f in range(self.num_filters):
            f_map = []
            for t in range(seq_len - self.kernel_size + 1):
                window = x[t : t + self.kernel_size]
                conv_val = np.sum(window * self.W[f]) + self.b[f]
                f_map.append(max(0.0, float(conv_val))) # ReLU
            feature_maps.append(max(f_map) if f_map else 0.0) # 1D Max-over-time pooling

        pooled = np.array(feature_maps) # (num_filters,)
        out = np.dot(pooled, self.W_proj) + self.b_proj
        return out


# ==========================================
# Vanilla Recurrent Neural Network
# ==========================================

class VanillaRNN:
    """Step-by-step unrolled Vanilla RNN cell."""

    def __init__(self, in_features: int, hidden_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim

        self.W_xh = rng.randn(in_features, hidden_dim) * np.sqrt(2.0 / in_features)
        self.W_hh = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_h = np.zeros((1, hidden_dim))

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Single recurrence step: h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)."""
        act = np.dot(x_t, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
        return np.tanh(act)

    def forward(self, x_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x_seq: (seq_len, batch_size, in_features)
        Returns: (outputs: (seq_len, batch_size, hidden_dim), h_final)
        """
        seq_len, batch_size, _ = x_seq.shape
        h = np.zeros((batch_size, self.hidden_dim))
        outputs = []
        for t in range(seq_len):
            h = self.step(x_seq[t], h)
            outputs.append(h)
        return np.stack(outputs, axis=0), h


# ==========================================
# Long Short-Term Memory (LSTM) Network
# ==========================================

class LSTM:
    """Complete Long Short-Term Memory network with all four gates."""

    def __init__(self, in_features: int, hidden_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        concat_dim = in_features + hidden_dim
        scale = np.sqrt(2.0 / concat_dim)

        # 4 gates concatenated: [forget, input, candidate, output]
        self.W_gates = rng.randn(concat_dim, 4 * hidden_dim) * scale
        self.b_gates = np.zeros(4 * hidden_dim)
        # Initialize forget gate bias to 1.0 (standard initialization heuristic)
        self.b_gates[:hidden_dim] = 1.0

    def step(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes single LSTM step:
        f_t = sigmoid(W_f [x_t, h_{t-1}] + b_f)
        i_t = sigmoid(W_i [x_t, h_{t-1}] + b_i)
        c_cand = tanh(W_c [x_t, h_{t-1}] + b_c)
        c_t = f_t * c_{t-1} + i_t * c_cand
        o_t = sigmoid(W_o [x_t, h_{t-1}] + b_o)
        h_t = o_t * tanh(c_t)
        """
        concat = np.concatenate([x_t, h_prev], axis=-1)
        gates = np.dot(concat, self.W_gates) + self.b_gates
        H = self.hidden_dim

        f = sigmoid(gates[..., :H])
        i = sigmoid(gates[..., H : 2 * H])
        c_cand = np.tanh(gates[..., 2 * H : 3 * H])
        o = sigmoid(gates[..., 3 * H :])

        c_t = f * c_prev + i * c_cand
        h_t = o * np.tanh(c_t)
        return h_t, c_t

    def forward(self, x_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        x_seq: (seq_len, batch_size, in_features)
        Returns: (all_h, h_last, c_last)
        """
        seq_len, batch_size, _ = x_seq.shape
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))

        all_h = []
        for t in range(seq_len):
            h, c = self.step(x_seq[t], h, c)
            all_h.append(h)

        return np.stack(all_h, axis=0), h, c


# ==========================================
# Gated Recurrent Unit (GRU) Network
# ==========================================

class GRU:
    """Gated Recurrent Unit (GRU) with reset and update gates."""

    def __init__(self, in_features: int, hidden_dim: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        concat_dim = in_features + hidden_dim
        scale = np.sqrt(2.0 / concat_dim)

        # Update (z) and Reset (r) gates
        self.W_z = rng.randn(concat_dim, hidden_dim) * scale
        self.b_z = np.zeros(hidden_dim)

        self.W_r = rng.randn(concat_dim, hidden_dim) * scale
        self.b_r = np.zeros(hidden_dim)

        # Candidate state
        self.W_h = rng.randn(concat_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

    def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        concat = np.concatenate([x_t, h_prev], axis=-1)
        z_t = sigmoid(np.dot(concat, self.W_z) + self.b_z) # Update gate
        r_t = sigmoid(np.dot(concat, self.W_r) + self.b_r) # Reset gate

        # Candidate state using gated previous hidden
        concat_reset = np.concatenate([x_t, r_t * h_prev], axis=-1)
        h_cand = np.tanh(np.dot(concat_reset, self.W_h) + self.b_h)

        h_t = (1.0 - z_t) * h_prev + z_t * h_cand
        return h_t

    def forward(self, x_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_len, batch_size, _ = x_seq.shape
        h = np.zeros((batch_size, self.hidden_dim))
        all_h = []
        for t in range(seq_len):
            h = self.step(x_seq[t], h)
            all_h.append(h)
        return np.stack(all_h, axis=0), h


# ==========================================
# BiLSTM with Additive Sequence Attention
# ==========================================

class BiLSTMWithAttention:
    """Bidirectional LSTM with Bahdanau/Additive Attention pooling for answer grading."""

    def __init__(self, in_features: int, hidden_dim: int, out_dim: int = 1, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.hidden_dim = hidden_dim

        self.forward_lstm = LSTM(in_features, hidden_dim, seed=seed)
        self.backward_lstm = LSTM(in_features, hidden_dim, seed=seed + 1)

        # Additive attention parameters
        bi_dim = 2 * hidden_dim
        self.W_attn = rng.randn(bi_dim, hidden_dim) * np.sqrt(2.0 / bi_dim)
        self.b_attn = np.zeros(hidden_dim)
        self.v_attn = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)

        # Regression prediction head
        self.W_out = rng.randn(bi_dim, out_dim) * np.sqrt(2.0 / bi_dim)
        self.b_out = np.zeros(out_dim)

    def forward(self, x_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        x_seq: (seq_len, batch_size, in_features)
        Returns: (pred_score: (batch_size, out_dim), attn_weights: (batch_size, seq_len))
        """
        seq_len, batch_size, _ = x_seq.shape

        # Forward pass
        h_fwd, _, _ = self.forward_lstm.forward(x_seq)

        # Backward pass (reverse sequence order)
        x_rev = x_seq[::-1]
        h_bwd_rev, _, _ = self.backward_lstm.forward(x_rev)
        h_bwd = h_bwd_rev[::-1] # restore sequence order

        # Concatenate forward and backward representations: (seq_len, batch_size, 2*H)
        h_bi = np.concatenate([h_fwd, h_bwd], axis=-1)

        # Reshape to (batch_size, seq_len, 2*H)
        h_bi_trans = np.transpose(h_bi, (1, 0, 2))

        # Additive Attention: e_t = v^T tanh(W_a h_t + b_a)
        u = np.tanh(np.dot(h_bi_trans, self.W_attn) + self.b_attn) # (B, T, H)
        scores = np.squeeze(np.dot(u, self.v_attn), axis=-1) # (B, T)
        alpha = softmax(scores, axis=-1) # (B, T)

        # Context vector: c = sum_t alpha_t h_t
        context = np.sum(h_bi_trans * np.expand_dims(alpha, axis=-1), axis=1) # (B, 2*H)

        # Output prediction
        out = np.dot(context, self.W_out) + self.b_out
        return out, alpha
