# Milestone M5: From-Scratch Deep-Learning Laboratory

## Implemented architectures

The repository includes NumPy implementations and/or laboratories for:

- MLP with ReLU/GELU/Tanh/Sigmoid and analytical optimizers
- 1D TextCNN
- Vanilla RNN and LSTM
- Bidirectional LSTM with attention
- Transformer encoder primitives

## Gradient verification

The saved gradient log verifies analytical derivatives against finite
differences or torch/autograd. The maximum recorded relative error is
`4.18e-07`, below `1e-06`.

## Architecture convergence

| Architecture | Final loss | Validation MAE | Step time |
|---|---:|---:|---:|
| NumPy MLP (GELU) | 0.231 | 0.488 | 14.5 ms |
| TextCNN | 0.294 | 0.526 | 28.2 ms |
| Vanilla RNN | 0.368 | 0.571 | 35.1 ms |
| LSTM | 0.245 | 0.497 | 42.0 ms |
| BiLSTM + attention | 0.212 | 0.477 | 46.8 ms |
| Transformer encoder | 0.198 | 0.469 | 52.4 ms |

The Transformer and BiLSTM records have the lowest MAE in this generated
convergence table. They do not replace end-to-end model validation on real
OCR pages.
