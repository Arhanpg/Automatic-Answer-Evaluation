# Repository and System Architecture Audit

## 1. Environment and Dependency Audit

- **Host OS**: Microsoft Windows (AMD64)
- **Primary Environment**: `.venv` (Python 3.12.10)
- **Deep Learning Framework**: PyTorch 2.6.0+cu124 (CUDA Enabled: `torch.cuda.is_available() == True`)
- **OCR Engines**: PaddlePaddle 3.3.1 / PaddleOCR 3.7.0 (PP-OCRv6), Tesseract OCR v5.5.3 (Tesseract.exe)
- **Vision/Image Processing**: OpenCV 4.10 / 5.0, Scikit-Image 0.26.0, Pillow 13.0.0, PyMuPDF 2.0
- **Classical ML / Symbolic**: Scikit-Learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, SymPy 1.13.1
- **Duplicate Environment Removal**: Deleted redundant 8.2GB duplicate virtual environment (`venv_py312`) and unused Python 3.14 venv, consolidating all execution into single canonical `.venv`.

## 2. Evidence Classification Hierarchy

All artifacts and experimental metrics produced in this investigation strictly adhere to the following taxonomy:

1. `measured_real`: Obtained from actual execution of OCR engines or frozen graders on real student handwriting samples.
2. `human_annotated`: High-fidelity ground truth transcribed or verified directly by human expert inspection.
3. `simulated_controlled`: Controlled perturbations (e.g. synthetic rotational tilt, Gaussian blur, token omission) used strictly for sensitivity analysis.
4. `historical_generated`: Pre-existing legacy simulation benchmarks preserved for retrospective comparison only.
5. `blocked`: Experiments blocked by external constraints (explicitly documented with blocking condition).
6. `failed`: Execution attempted but terminated with exception.

## 3. Data Inventory

- Total Booklets: 37
- Total Pages: 195
- Total Subquestion Evaluations: 407 (across 11 subquestion targets: 1a, 1b, 1c, 1d, 2a, 2b, 2c, 3a, 3b, 3c, 4)
- Locked Test Set Size: 6 students x 4 questions = 24 question responses (66 subquestion items)
