"""
src/from_scratch/mlp_numpy.py - Milestone M5: Pure NumPy Multi-Layer Perceptron

Features:
- Pure NumPy forward and analytical backpropagation
- Activations: ReLU, GELU (tanh approximation), Tanh, Sigmoid
- Losses: MSE, MAE, Huber Loss (smooth L1)
- Optimizers: SGD, SGD with Momentum, RMSProp, Adam
- Exact analytical gradient computation for mathematical verification (< 10^-6)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


# ==========================================
# Activation Functions & Derivatives
# ==========================================

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.input > 0).astype(np.float64)


class GELU:
    """Gaussian Error Linear Unit with tanh approximation."""
    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
        return 0.5 * x * (1.0 + np.tanh(inner))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        x = self.input
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
        tanh_val = np.tanh(inner)
        sech2 = 1.0 - np.power(tanh_val, 2)
        dinner_dx = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
        dgelu_dx = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * dinner_dx
        return grad_output * dgelu_dx


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1.0 - np.power(self.output, 2))


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.output * (1.0 - self.output)


# ==========================================
# Loss Functions & Derivatives
# ==========================================

class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return float(np.mean(np.power(y_pred - y_true, 2)))

    def backward(self) -> np.ndarray:
        n = self.y_pred.size
        return (2.0 / n) * (self.y_pred - self.y_true)


class MAELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return float(np.mean(np.abs(y_pred - y_true)))

    def backward(self) -> np.ndarray:
        n = self.y_pred.size
        return (1.0 / n) * np.sign(self.y_pred - self.y_true)


class HuberLoss:
    """Huber Loss (Smooth L1 Loss)."""
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        diff = np.abs(y_pred - y_true)
        quadratic = np.minimum(diff, self.delta)
        linear = diff - quadratic
        loss = 0.5 * np.power(quadratic, 2) + self.delta * linear
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        diff = self.y_pred - self.y_true
        n = self.y_pred.size
        grad = np.where(np.abs(diff) <= self.delta, diff, self.delta * np.sign(diff))
        return grad / n


# ==========================================
# Dense Linear Layer
# ==========================================

class LinearLayer:
    def __init__(self, in_features: int, out_features: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        # He / Xavier initialization
        limit = np.sqrt(2.0 / in_features)
        self.W = rng.randn(in_features, out_features) * limit
        self.b = np.zeros((1, out_features))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # grad_output: (batch_size, out_features)
        # self.x: (batch_size, in_features)
        self.dW = np.dot(self.x.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)


# ==========================================
# Optimizers
# ==========================================

class SGD:
    def __init__(self, params: List[Tuple[np.ndarray, np.ndarray]], lr: float = 0.01):
        self.params = params # List of (param, grad)
        self.lr = lr

    def step(self):
        for p, grad in self.params:
            p -= self.lr * grad


class SGDMomentum:
    def __init__(self, params: List[Tuple[np.ndarray, np.ndarray]], lr: float = 0.01, beta: float = 0.9):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.velocities = [np.zeros_like(p) for p, _ in params]

    def step(self):
        for i, (p, grad) in enumerate(self.params):
            self.velocities[i] = self.beta * self.velocities[i] + self.lr * grad
            p -= self.velocities[i]


class RMSProp:
    def __init__(self, params: List[Tuple[np.ndarray, np.ndarray]], lr: float = 0.001, alpha: float = 0.99, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.sq_grads = [np.zeros_like(p) for p, _ in params]

    def step(self):
        for i, (p, grad) in enumerate(self.params):
            self.sq_grads[i] = self.alpha * self.sq_grads[i] + (1.0 - self.alpha) * np.power(grad, 2)
            p -= (self.lr / (np.sqrt(self.sq_grads[i]) + self.eps)) * grad


class Adam:
    def __init__(self, params: List[Tuple[np.ndarray, np.ndarray]], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (p, grad) in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * np.power(grad, 2)
            # Bias correction
            m_hat = self.m[i] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[i] / (1.0 - (self.beta2 ** self.t))
            p -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


# ==========================================
# End-to-End MLP Architecture
# ==========================================

class NumpyMLP:
    """Configurable Multi-Layer Perceptron built from scratch in NumPy."""

    def __init__(self, layer_sizes: List[int], activation: str = "gelu", seed: int = 42):
        self.layers: List[LinearLayer] = []
        self.activations = []
        self.act_type = activation.lower()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i + 1], seed=seed + i))
            if i < len(layer_sizes) - 2:
                if self.act_type == "relu":
                    self.activations.append(ReLU())
                elif self.act_type == "gelu":
                    self.activations.append(GELU())
                elif self.act_type == "tanh":
                    self.activations.append(Tanh())
                elif self.act_type == "sigmoid":
                    self.activations.append(Sigmoid())
                else:
                    self.activations.append(ReLU())

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for i, layer in enumerate(self.layers):
            out = layer.forward(out)
            if i < len(self.activations):
                out = self.activations[i].forward(out)
        return out

    def backward(self, grad_loss: np.ndarray):
        grad = grad_loss
        for i in reversed(range(len(self.layers))):
            if i < len(self.activations):
                grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)

    def get_parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        params = []
        for l in self.layers:
            params.append((l.W, l.dW))
            params.append((l.b, l.db))
        return params
