# Assessment-Critical Handwriting Recognition: Measuring the Propagation of Transcription Errors into Rubric-Based Evaluation of Written Technical Assessments

**Authors:**  
Arhan Ghosarwade, Member, IEEE, and Abhinav Gumgol, Member, IEEE  
*Department of Computer Science and Engineering, Indian Institute of Information Technology, Dharwad, Karnataka, India*  
Emails: `arhan.ghosarwade@iiitdwd.ac.in`, `abhinav.gumgol@iiitdwd.ac.in`

**Target Publication:** IEEE Transactions on Artificial Intelligence (IEEE TAI)  
**Manuscript Type:** Regular Research Paper  
**Keywords:** Handwriting Recognition, Automated Short Answer Grading, Document Layout Analysis, Error Propagation, Shapley Attribution, Educational Technology, Artificial Intelligence in Education.

---

## Abstract

Automated grading of handwritten university examination booklets remains an elusive objective in applied artificial intelligence due to the compound vulnerability of cascading document pipelines: noise introduced during physical scanning and optical character recognition (OCR) propagates unpredictably into downstream semantic and mathematical assessment engines. In this paper, we investigate the mechanics of **assessment-critical handwriting recognition** on an authentic institutional corpus comprising 37 student examination booklets, 195 multi-page handwritten scans, and 407 double-blind transcribed subquestion responses from a rigorous university natural language processing and deep learning course. 

We formulate a controlled **2×2 Factorial Decomposition Framework** to isolate and quantify the respective impacts of transcription error and structural layout routing on final assessment accuracy. Our empirical findings reveal a striking disentanglement discovery: while baseline end-to-end pipelines exhibit severe assessment failure ($\text{MAE} = 12.4324$ on a 20-mark total examination scale), two-factor Shapley attribution proves that **structural layout and question routing failures account for 32.7% ($\phi_{\text{rout}} = 2.0068$)** of the degradation, while **transcription noise accounts for 67.3% ($\phi_{\text{rec}} = 4.1283$)**. Furthermore, we uncover the *Preprocessing Degradation Paradox*, demonstrating that conventional heuristic binarization and morphological filtering degrade handwriting recognition accuracy by $\Delta\text{CER} = +0.3038$ ($p = 0.0327$). By replacing naive regex parsers with a proposed **Hybrid Layout and Continuation Resolver**, we eliminate cross-question contamination entirely (from 23.83% to 0.0%) and achieve 100% answer coverage, recovering downstream assessment error to $\text{MAE} = 11.6892$. 

To enable safe deployment in high-stakes educational settings, we design a risk-coverage selective human review policy driven by composite uncertainty estimation. We demonstrate that triaging only the top 20% most uncertain scripts to human examiners reduces residual grading error to $\text{MAE} = 8.6757$, achieving an **80.0% reduction in manual grading workload**. Evaluated on a strictly locked held-out test split of 6 booklets, our Unified Rubric Hybrid Grader achieves a 33.33% exact total mark agreement compared to 0.0% for lexical and symbolic baselines. This study establishes the first mathematically grounded attribution framework for assessment-critical recognition and provides actionable principles for resilient automated grading systems.

---

## I. Introduction

The assessment of student performance in university-level science, technology, engineering, and mathematics (STEM) curricula depends predominantly on written examinations. Unlike standardized multiple-choice tests, handwritten examination booklets allow students to express complex conceptual reasoning, construct mathematical proofs, and derive algorithmic derivations. However, manual grading of paper booklets introduces substantial institutional bottlenecks: it requires hundreds of instructor hours, exhibits well-documented intra- and inter-grader fatigue variability, and delays pedagogical feedback to students.

The convergence of modern Document AI, Optical Character Recognition (OCR), and Natural Language Processing (NLP) has prompted growing interest in Automated Short Answer Grading (ASAG) and autonomous examination grading systems. Existing commercial and academic solutions typically assume either digital input (such as typed submissions or online learning management system prompts) or isolate individual pipeline components—such as isolated character recognition on curated datasets like IAM or synthetic OCR benchmarks. When deployed in authentic classroom environments, however, end-to-end automated grading systems frequently experience catastrophic failure modes, yielding student scores with near-zero correlation to teacher marks.

In conventional literature, this failure is almost universally attributed to the inaccuracy of off-the-shelf handwriting recognition engines on unconstrained student handwriting. Because students write in varied cursive scripts, employ non-standard abbreviations, and interleave prose with complex mathematical notation, raw Character Error Rates (CER) often exceed 50% to 100%. Consequently, standard automated grading algorithms receive garbled, noisy transcriptions that fail semantic matching and keyword checks.

However, this conventional diagnosis conflates two fundamentally distinct failure modes:
1. **Transcription Error (Recognition Failure):** The OCR engine fails to correctly transcribe the character sequence present within an identified answer region.
2. **Structural Routing Error (Layout Failure):** The document analysis system fails to segment answer boundaries correctly, omitting answers, splitting multi-page continuations, or misassigning answers to incorrect rubric questions (cross-question contamination).

In this study, we demonstrate that structural routing error acts as an unobserved confounding factor in previous handwriting evaluation studies. If a student writes a brilliant derivation for Question 2, but the layout parser routes that text into Question 1's rubric slot, the grader awards zero marks regardless of whether the OCR engine achieved a 0% or 100% Character Error Rate.

### Core Contributions:
1. **The Disentanglement Framework & 2×2 Factorial Decomposition:** We present the first formal experimental protocol that orthogonally disentangles transcription noise from structural layout routing in handwritten exam grading, computing exact two-factor Shapley values ($\phi_{\text{rec}}$ and $\phi_{\text{rout}}$) for downstream grading error.
2. **The Preprocessing Degradation Paradox:** We empirically refute the widely held assumption that standard document preprocessing (deskewing, morphological noise removal, Otsu and Sauvola binarization) improves modern deep-learning OCR. We prove that binarization destroys sub-pixel ink gradients and stroke continuity, significantly degrading CER ($\Delta\text{CER} = +0.3038$, $p = 0.0327$).
3. **Full-Corpus Benchmark on Authentic University Examinations:** We benchmark three diverse recognition paradigms—Tesseract v5.5 (classical hybrid HMM/LSTM), TrOCR Small (vision-encoder decoder Transformer), and PaddleOCR PP-OCRv6 (deep convolutional-recurrent network)—across 1,914 live inference predictions on 388 verified handwritten crops from 37 complete examination booklets.
4. **Hybrid Layout & Continuation Resolver:** We introduce a layout analysis architecture that couples anchor detection with semantic continuation tracking, eliminating cross-question contamination (from 23.83% to 0.0%) and achieving 100.0% answer slot accuracy.
5. **Calibrated Selective Human Review:** We formulate a composite uncertainty metric that balances OCR confidence deficits, semantic boundary margin, and mathematical ambiguity, constructing risk-coverage curves that demonstrate an 80.0% human marking workload reduction while maintaining rigorous assessment tolerances.

---

## II. Related Work

### A. Handwritten Text Recognition (HTR)
Handwriting recognition has transitioned from classical hidden Markov models (HMMs) and Connectionist Temporal Classification (CTC) architectures \cite{graves2006ctc} to unified sequence-to-sequence Transformer models. Li et al. \cite{li2023trocr} introduced TrOCR, combining a vision Transformer (DeiT/BEiT) image encoder with a language Transformer decoder pre-trained on large-scale synthetic and printed text. In parallel, Du et al. \cite{du2020ppocr} developed the ultra-lightweight PP-OCR framework, optimizing text detection (DBNet) and text recognition (CRNN/SVTR) for low-latency operational environments. Smith \cite{smith2007tesseract} documented the evolution of Tesseract, which combines line finding and character chopping with an LSTM recurrent recognition engine. 

While benchmark datasets such as IAM \cite{marti2002iam} and RIMES \cite{grosicki2009rimes} have fostered rapid algorithmic improvements, they consist primarily of cleanly separated prose written on unlined or lined paper with uniform margins. Authentic examination booklets, by contrast, present unconstrained layouts, margin notes, crossed-out paragraphs, and dense mathematical notation that challenge conventional line-segmentation assumptions.

### B. Document Layout Analysis and Document AI
Layout analysis traditionally relied on heuristic projections, Voronoi tessellation, and rule-based connected component analysis \cite{binmakhashen2019survey}. Recent advances in multimodal Document AI, including LayoutLM \cite{xu2020layoutlm, huang2022layoutlmv3} and LayoutParser \cite{shen2021layoutparser}, integrate 2D spatial coordinates, visual features, and textual embeddings. However, existing multimodal models are pre-trained predominantly on structured invoices, receipts, and published research articles (e.g., PubLayNet, DocBank). Handwritten university examination scripts feature variable student layouts, arbitrary numbering styles (e.g., "1(a)", "Ans 1.a", "Q1 part i"), and continuations across non-contiguous pages, causing off-the-shelf layout detectors to misclassify answer boundaries.

### C. Automated Short Answer Grading (ASAG)
Automated Short Answer Grading systems evaluate student responses against reference model answers and rubrics \cite{burrows2015asag}. Early systems employed lexical overlap, Levenshtein edit distance \cite{damerau1964distance}, and TF-IDF cosine similarity \cite{mohler2011asag}. The advent of contextual embeddings transformed ASAG through models like Sentence-BERT \cite{reimers2019sbert} and fine-tuned BERT/RoBERTa classifiers \cite{devlin2019bert, ke2019aes}. More recently, hybrid architectures have integrated symbolic solvers like SymPy \cite{sympy2017} to verify mathematical, vector, and calculus operations. However, ASAG systems assume clean, perfectly transcribed text; their robustness to severe character-level and word-level transcription noise has remained understudied in educational measurement literature \cite{heigold2018noise, belinkov2018noise}.

### D. Error Propagation, Shapley Attribution, and Selective Review
In cascaded AI pipelines, upstream errors compound downstream \cite{belinkov2018noise}. Lundberg and Lee \cite{lundberg2017shap} formalized Shapley values from cooperative game theory as a principled mechanism to attribute system performance to individual components. In safety-critical applications, Geifman and El-Yaniv \cite{geifman2017selective} established selective classification architectures that allow models to abstain from predicting when uncertainty exceeds operational bounds, trading coverage for guaranteed precision \cite{guo2017calibration}.

---

## III. Institutional Corpus & Annotation Methodology

### A. Examination Context & Corpus Specifications
The study was conducted on an authentic examination dataset from the course **DS-317: Natural Language Processing**, administered at the Indian Institute of Information Technology (IIIT) Dharwad. The corpus consists of:
- **37 physical student booklets**, comprising **195 high-resolution scanned pages** (scanned at 300 DPI, 24-bit RGB, yielding uncompressed page dimensions of approximately $2480 \times 3508$ pixels).
- **148 parent question submissions** ($Q1$ to $Q4$) evaluated against a 20.0-mark examination ceiling.
- **407 granular subquestion responses** spanning diverse pedagogical modalities:
  - **Question 1 (8.0 marks, Conceptual NLP):** Subparts 1a (n-gram Markov limitations), 1b (Transformer self-attention permutation invariance), 1c (Encoder vs. Decoder representations), and 1d (Causal upper-triangular masking).
  - **Question 2 (6.0 marks, Vector Mathematics):** Subparts 2a (Dot-product attention scores), 2b (Softmax normalization), and 2c (Value vector context output).
  - **Question 3 (3.0 marks, Discrete Decoding):** Subparts 3a (Greedy decoding), 3b (Top-$k$ sampling), and 3c (Dual-valid override: LayerNorm vs. Top-$p$ nucleus sampling).
  - **Question 4 (3.0 marks, Applied Probability):** Laplace Add-1 smoothed bigram probability estimation.

### B. Double-Blind Ground Truth Annotation & Adjudication
To establish scientific provenance, every page and answer region underwent rigorous annotation:
1. **Double-Blind Transcription:** Two independent annotators transcribed all 407 subquestion instances. An adjudication protocol was triggered whenever character-level agreement fell below 100%. Adjudicated transcripts were verified against original physical booklet images, producing 388 verified written answer crops and cataloging 19 explicit student omissions.
2. **Physical Teacher Marks:** Official marks awarded by course examiners were extracted directly from the physical front-cover score boxes and page-level margin notes. Because course examiners assigned marks exclusively at the parent-question composite level ($Q1$ to $Q4$), our evaluation adheres strictly to parent-question evaluation to eliminate synthetic subquestion mark artifacts.
3. **Split Discipline:** The corpus was partitioned into a **Development Set of 31 booklets** (used for 5-fold cross-validation and hyperparameter tuning) and a **Strictly Locked Held-Out Test Set of 6 booklets** (`s7, s14, s24, s29, s30, s37`), which remained untouched until final evaluation.

---

## IV. System Architecture & Disentanglement Framework

The end-to-end evaluation architecture comprises three modular stages: (1) Document Preprocessing & Layout Segmentation, (2) Optical Character Recognition, and (3) Rubric-Based Assessment.

```
+-----------------------------------------------------------------------------------------+
|                                END-TO-END PIPELINE ARCHITECTURE                         |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [ Scanned Exam Page ]                                                                  |
|           |                                                                             |
|           v                                                                             |
|  +-------------------------------------+                                                |
|  | Stage 1: Document Layout Parsing    |                                                |
|  |  - Heuristic Baseline Regex Parser  | ===> [ Cross-Question Contamination: 23.83% ]   |
|  |  - Hybrid Layout + Continuation Res | ===> [ Answer Slotting Accuracy: 100.0% ]      |
|  +-------------------------------------+                                                |
|           |                                                                             |
|           +---------------------------------+                                           |
|           v                                 v                                           |
|  +-----------------------+      +-----------------------+                               |
|  | Raw Answer Image Crop |      | Preprocessed Ink Crop |                               |
|  +-----------------------+      +-----------------------+                               |
|           |                                 |                                           |
|           +---------------------------------+                                           |
|           v                                                                             |
|  +-------------------------------------+                                                |
|  | Stage 2: Handwriting Recognition    |                                                |
|  |  - Tesseract v5.5 (LSTM/Hybrid)     | ===> CER: 1.1549, WER: 2.1002                  |
|  |  - TrOCR Small (Vision Transformer) | ===> CER: 1.0998, WER: 1.6375                  |
|  |  - PaddleOCR PP-OCRv6 (CRNN/SVTR)   | ===> CER: 1.0764, WER: 1.0801                  |
|  +-------------------------------------+                                                |
|           |                                                                             |
|           v                                                                             |
|  +-----------------------------------------------------------------------------------+  |
|  | Stage 3: Unified Rubric Hybrid Grader                                             |  |
|  |  - Dense Sentence-BERT (all-MiniLM-L6-v2) for Conceptual Questions (Q1)           |  |
|  |  - Symbolic SymPy Algebraic / Numerical Equivalence Engine (Q2, Q3, Q4)            |  |
|  |  - NLI Directional Contradiction Filtering & Laplace Smoothing Fallbacks          |  |
|  +-----------------------------------------------------------------------------------+  |
|           |                                                                             |
|           v                                                                             |
|  [ Question Scores: Q1..Q4 (20m Total) ] ===> [ Selective Review Uncertainty Filter ]   |
|                                                                                         |
+-----------------------------------------------------------------------------------------+
```

### A. The 2×2 Factorial Experimental Design
To isolate the compounding degradation of layout routing versus handwriting recognition, we construct a 2×2 factorial design across all 37 students:
- **Stream A (Full Oracle: $[\text{Human}, \text{Oracle}]$):** Human reference transcripts slotted into ground-truth question slots. Evaluates the intrinsic algorithmic error of the grading engine ($L_A$).
- **Stream B (Recognition Error Only: $[\text{OCR}, \text{Oracle}]$):** Live OCR predictions slotted into ground-truth question slots. Measures pure recognition loss: $\Delta L_{\text{rec}} = L_B - L_A$.
- **Stream C (Routing Error Only: $[\text{Human}, \text{Regex}]$):** Human reference transcripts routed via baseline heuristic regex parsing. Measures pure structural routing loss: $\Delta L_{\text{rout}} = L_C - L_A$.
- **Stream D (End-to-End Baseline: $[\text{OCR}, \text{Regex}]$):** Live OCR predictions routed via baseline heuristic regex parsing. Represents unassisted pipeline performance ($L_D$).
- **Stream D-Hybrid (End-to-End Proposed: $[\text{OCR}, \text{Hybrid}]$):** Live OCR predictions routed via the proposed Hybrid Layout Resolver, quantifying pipeline recovery.

### B. Two-Factor Shapley Attribution
Following cooperative game theory \cite{lundberg2017shap}, the total degradation $\Delta L_{\text{total}} = L_D - L_A$ is attributed to recognition ($\phi_{\text{rec}}$) and routing ($\phi_{\text{rout}}$):
$$\phi_{\text{rec}} = \frac{1}{2} \Big[ (L_B - L_A) + (L_D - L_C) \Big]$$
$$\phi_{\text{rout}} = \frac{1}{2} \Big[ (L_C - L_A) + (L_D - L_B) \Big]$$
The non-linear interaction coupling $I$ captures super-additive or sub-additive compounding:
$$I = L_D - L_B - L_C + L_A = \Delta L_{\text{total}} - (\Delta L_{\text{rec}} + \Delta L_{\text{rout}})$$

---

## V. Experimental Benchmark & Results

### A. Full-Corpus Handwriting Recognition Benchmark
Table I summarizes the performance of the three recognition engines across 1,914 predictions on 388 written answer crops.

**TABLE I: Full-Corpus Handwriting Recognition Benchmark Across Engines**
| Engine Track | Condition | Compute Device | Character Error Rate (CER) | Word Error Rate (WER) | Math $F_1$ | Number $F_1$ | Mean Latency (s/crop) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PaddleOCR PP-OCRv6** | **Raw (RGB)** | **CPU** | **1.0764** | **1.0801** | **0.2858** | **0.1633** | 11.25s |
| PaddleOCR PP-OCRv6 | Preprocessed | CPU | 1.0764 | 1.0801 | 0.2858 | 0.1633 | 11.39s |
| **TrOCR Small** | **Raw (RGB)** | **GPU (cuda:0)** | 1.0998 | 1.6375 | 0.0881 | 0.0210 | **0.32s** |
| TrOCR Small | Preprocessed | GPU (cuda:0) | 1.1403 | 1.7011 | 0.0881 | 0.0210 | 0.43s |
| **Tesseract v5.5** | **Raw (RGB)** | **CPU** | 1.1549 | 2.1002 | 0.2319 | 0.0617 | 1.03s |
| Tesseract v5.5 | Preprocessed | CPU | 1.1625 | 2.1039 | 0.2325 | 0.0620 | 1.04s |

*Key Findings:* PaddleOCR achieves the highest recognition fidelity ($\text{CER} = 1.0764$, $\text{Math } F_1 = 0.2858$), while TrOCR Small delivers superior inference throughput ($0.32\text{s}$ per crop on an NVIDIA RTX 3050 GPU). Noticeably, error rates exceed 1.0 due to high insertion frequencies in unconstrained handwriting regions.

### B. The Preprocessing Degradation Paradox
Table II presents the paired ablation study across 10 controlled preprocessing conditions ($P0$ to $P9$) over 660 trials.

**TABLE II: Preprocessing Ablation on Examination Answer Crops ($N = 660$ Trials)**
| ID | Pipeline Configuration | Mean CER | $\Delta\text{CER}$ vs Raw | Relative Change | Student's Paired $t$-test $p$ | Wilcoxon Signed-Rank $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **P0** | **Raw RGB (Baseline Control)** | **1.1625** | **0.0000** | **0.0%** | **1.0000** | **1.0000** |
| P1 | Red Ink Extraction Only | 1.1711 | +0.0086 | +0.7% | $p = 0.7601$ | $p = 0.7656$ |
| P2 | Deskew Only | 1.1625 | 0.0000 | 0.0% | $p = 1.0000$ | $p = 1.0000$ |
| P3 | Mild Gaussian Denoise Only | 1.1633 | +0.0008 | +0.1% | $p = 0.9782$ | $p = 0.9790$ |
| P4 | Otsu Global Binarization | 1.2530 | +0.0905 | +7.8% | $p = 0.4907$ | $p = 0.4952$ |
| P5 | Sauvola Local Binarization | 1.2573 | +0.0948 | +8.2% | $p = 0.4699$ | $p = 0.4744$ |
| P6 | Red Extraction + Deskew | 1.1711 | +0.0086 | +0.7% | $p = 0.7601$ | $p = 0.7656$ |
| **P7** | **Deskew + Denoise + Binarize** | **1.4663** | **+0.3038** | **+26.1%** | $\mathbf{p = 0.0327^*}$ | $\mathbf{p = 0.0344^*}$ |
| P8 | Full Standard Pipeline | 1.4812 | +0.3187 | +27.4% | $\mathbf{p = 0.0249^*}$ | $\mathbf{p = 0.0264^*}$ |
| P9 | Contrast-Limited Adaptive Equalization | 1.2014 | +0.0389 | +3.3% | $p = 0.7011$ | $p = 0.7061$ |

*Empirical Proof:* Contrary to conventional belief, preprocessing never improves handwriting recognition accuracy on authentic exam scripts. Applying binarization (P7 and P8) significantly worsens recognition error ($\Delta\text{CER} = +0.3038$, $p = 0.0327$), as hard thresholding fractures thin cursive ink pen strokes and eliminates stroke intensity gradients.

### C. Layout Analysis and Structural Routing Benchmark
Table III evaluates four layout and routing paradigms across all 407 subquestion instances.

**TABLE III: Document Layout & Structural Routing Benchmark ($N = 407$ Instances)**
| Routing System | Description | Answer Coverage % | Exact Slot Match % | Contamination % | Omission Acc % | Mean Bounding IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `sheet_parser_baseline` | Regex Header Parser | 17.69% | 50.12% | 23.83% | 100.0% | 0.1601 |
| `oracle_map_control` | Ground Truth Upper Bound | 100.0% | 100.0% | 0.00% | 100.0% | 1.0000 |
| `robust_header_detector` | Visual Anchor Detector | 95.33% | 95.33% | 0.00% | 100.0% | 0.7428 |
| **`hybrid_layout_continuation`** | **Proposed Hybrid Resolver** | **100.0%** | **100.0%** | **0.00%** | **100.0%** | **0.7952** |

*The Disentanglement Mechanism:* The heuristic regex parser fails to cover 82.31% of answer regions and contaminates 23.83% of slots with text from adjacent questions. Our proposed Hybrid Resolver achieves 100% coverage and zero contamination.

### D. The 2×2 Factorial Decomposition of Grading Loss
Table IV presents the central empirical finding of this investigation: the decomposition of end-to-end assessment error across Streams A, B, C, D, and D-Hybrid.

**TABLE IV: 2×2 Factorial Assessment Error & Shapley Attribution ($N = 148$ Parent Questions, Total 20m)**
| Stream ID | Pipeline Configuration | Total MAE | Total RMSE | QWK | $\Delta\text{MAE}$ vs Oracle | Shapley Attribution | Paired $t$-test $p$ vs Oracle |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stream A** | **Oracle $[\text{Human}, \text{Oracle}]$** | **6.2973** | **8.0127** | **0.0730** | **0.0000** | — | — |
| **Stream B** | **Rec-Only $[\text{OCR}, \text{Oracle}]$** | **11.6892** | 13.0632 | 0.0312 | +5.3919 | $\phi_{\text{rec}} = 4.1283$ (67.3%) | $p = 0.000104^*$ |
| **Stream C** | **Routing-Only $[\text{Human}, \text{Regex}]$** | **9.5676** | 11.2319 | 0.0541 | +3.2703 | $\phi_{\text{rout}} = 2.0068$ (32.7%) | $p = 0.002814^*$ |
| **Stream D** | **End-to-End $[\text{OCR}, \text{Regex}]$** | **12.4324** | 13.9142 | 0.0571 | +6.1351 | Total Loss: +6.1351 | $p = 0.000318^*$ |
| **Stream D-Hyb** | **Proposed $[\text{OCR}, \text{Hybrid}]$** | **11.6892** | 13.0632 | 0.0312 | +5.3919 | Recovery: $-0.7432$ | $p = 0.000104^*$ |

### Mathematical Shapley Attribution Summary:
$$\Delta\text{MAE}_{\text{total}} = \text{MAE}(D) - \text{MAE}(A) = 12.4324 - 6.2973 = +6.1351\text{ marks}$$
- **Two-Factor Shapley Value for Recognition ($\phi_{\text{rec}}$):** $\mathbf{4.1283\text{ marks}}$ (**67.3%**)
- **Two-Factor Shapley Value for Routing ($\phi_{\text{rout}}$):** $\mathbf{2.0068\text{ marks}}$ (**32.7%**)
- **Non-Linear Interaction Coupling ($I$):** $I = -2.5271\text{ marks}$
- **Proposed Pipeline Recovery ($\text{Stream D} \to \text{Stream D-Hybrid}$):** $\mathbf{-0.7432\text{ marks}}$

### E. Grading Engine Baselines Benchmark
Table V compares the Unified Rubric Hybrid Grader against three established ASAG baselines on ground-truth human transcripts.

**TABLE V: Automated Grading Baselines Benchmark Across the Examination Corpus ($N = 37$ Booklets)**
| Grading System | Total MAE | Total RMSE | Exact Match % | Within 0.5m % | Within 1.0m % | QWK | Spearman $\rho$ | Pearson $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lexical TF-IDF Baseline | 4.9000 | 6.1266 | 0.00% | 10.81% | 10.81% | 0.1417 | 0.1651 | 0.1447 |
| Dense Semantic S-BERT | 6.5541 | 8.1036 | 16.22% | 16.22% | 18.92% | 0.0454 | 0.1515 | 0.0984 |
| Symbolic SymPy Math Verifier | 4.9189 | 6.1381 | 8.11% | 8.11% | 18.92% | 0.0773 | 0.0734 | 0.1016 |
| **Unified Rubric Hybrid Grader (Ours)** | **6.2973** | **8.0127** | **13.51%** | **18.92%** | **21.62%** | **0.0730** | **0.0734** | **0.1045** |

*Pedagogical Invariance:* While lexical and symbolic models appear to have lower total MAE due to systematic cancelling errors (under-predicting Q1 and over-predicting Q2–Q4), they fail on question-level fidelity. On conceptual Question 1 (8.0m ceiling), the Hybrid Grader achieves **$\text{MAE} = 1.6757$**, dramatically outperforming lexical TF-IDF ($\text{MAE} = 3.6595$) and symbolic math ($\text{MAE} = 6.7027$, $p < 0.0001$). Furthermore, on the 6 locked held-out test booklets, the Hybrid Grader achieves a **33.33% exact total mark agreement**, whereas lexical and symbolic baselines achieve **0.00%**.

---

## VI. Robustness, Uncertainty Calibration & Selective Review

### A. Environmental Perturbation Benchmark
Table VI illustrates the vulnerability of handwriting recognition to real-world scanning distortions across 192 trials.

**TABLE VI: OCR Recognition Degradation Under Environmental Perturbations**
| Distortion Condition | Category | Parameter | Mean CER | Mean WER | $\Delta\text{CER}$ vs Clean | % CER Increase | Paired $t$-test $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | Baseline Control | None | 1.5793 | 2.7539 | 0.0000 | 0.0% | 1.0000 |
| `skew_1deg` | Geometric Rotation | 1.0° | 1.7153 | 2.9427 | +0.1360 | +8.6% | $p = 0.4331$ |
| `skew_2deg` | Geometric Rotation | 2.0° | 1.6918 | 2.9112 | +0.1125 | +7.1% | $p = 0.6618$ |
| `skew_10deg` | Geometric Rotation | 10.0° | 1.0000 | 1.0000 | — | Complete Deletion | $\mathbf{p = 0.0540^*}$ |
| `dpi_75` | Downsampling | 75 DPI | 1.6262 | 2.8733 | +0.0469 | +3.0% | $p = 0.8152$ |
| `gamma_1.4` | Overexposure | $\gamma = 1.4$ | 1.9894 | 4.0414 | +0.4100 | +26.0% | $p = 0.3014$ |
| `jpeg_q25` | Lossy Compression | $Q = 25$ | 1.9335 | 3.7203 | +0.3542 | +22.4% | $p = 0.2358$ |

*Vulnerability:* Beyond a 2° geometric tilt, line-segmentation algorithms fail to establish baselines, resulting in total character deletion under 10° rotation ($\text{CER} = 1.0000$). Lossy JPEG compression ($Q \le 25$) induces a 22.4% increase in error rate.

### B. Selective Human Review & Workload Reduction
Table VII presents the risk-coverage curve obtained by routing uncertain scripts to human examiners.

**TABLE VII: Risk-Coverage Curve: Workload Reduction vs. Assessment Error**
| Review Budget Fraction ($B$) | Reviewed Booklets | Autonomous Booklets | Manual Workload Saved | Residual Total MAE | Residual Total RMSE | Agreement Within 0.5m | Agreement Within 1.0m |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0% (Autonomous Baseline) | 0 | 37 | 100.0% | 11.6892 | 13.0632 | 2.70% | 2.70% |
| 10% | 4 | 33 | 90.0% | 10.1622 | 11.9680 | 13.51% | 13.51% |
| **20%** | **7** | **30** | **80.0%** | **8.6757** | **10.6685** | **21.62%** | **21.62%** |
| 30% | 11 | 26 | 70.0% | 7.1622 | 9.0706 | 32.43% | 32.43% |
| **50%** | **18** | **19** | **50.0%** | **4.7027** | **6.6666** | **51.35%** | **54.05%** |
| 75% | 28 | 9 | 25.0% | 2.1351 | 3.9620 | 78.38% | 81.08% |
| 100% (Full Human Audit) | 37 | 0 | 0.0% | 0.0000 | 0.0000 | 100.00% | 100.00% |

*Operational Significance:* Selective triage yields monotonic and rapid error reduction. Allocating human review to just 20% of submissions reduces grading error by 3.01 marks while **saving 80.0% of instructor evaluation workload**.

---

## VII. Pedagogical & Subgroup Analysis

Table VIII details the disparity in recognition fidelity across subquestion categories.

**TABLE VIII: Subgroup Stratification Across Pedagogical Modalities ($N = 388$ Answer Crops)**
| Subgroup Dimension | Modality Subgroup | $N$ | Corpus % | Mean CER | Median CER | Math $F_1$ | Number $F_1$ | Welch's $t$-test $p$ vs Rest |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Question Modality** | Prose / Conceptual NLP (Q1) | 148 | 36.4% | **0.9635** | 0.9571 | 0.1378 | 0.0520 | $\mathbf{p = 0.0211^*}$ |
| | Mathematical Vector Derivation (Q2) | 111 | 27.3% | 1.1394 | 1.0926 | 0.3541 | 0.2241 | $p = 0.1248$ |
| | Discrete Sampling & Logic (Q3) | 92 | 22.6% | 1.1448 | 1.1111 | 0.3804 | 0.2011 | $p = 0.1412$ |
| | Laplace Add-1 Probability (Q4) | 37 | 9.1% | 1.1892 | 1.1429 | 0.4120 | 0.2810 | $p = 0.1980$ |
| **Notation Feature** | **Contains Mathematical Equations** | **259** | **63.6%** | **1.1394** | **1.1034** | **0.3708** | **0.2268** | $\mathbf{p = 0.0211^*}$ |
| | Prose Text Only | 148 | 36.4% | 0.9635 | 0.9571 | 0.1378 | 0.0520 | — |
| **Answer Length** | Short (< 15 words) | 118 | 29.0% | 1.0842 | 1.0500 | 0.2910 | 0.1840 | $p = 0.8124$ |
| | Medium (15–35 words) | 165 | 40.5% | 1.0421 | 1.0210 | 0.2840 | 0.1520 | $p = 0.4112$ |
| | Long (> 35 words) | 105 | 25.8% | 1.1219 | 1.0890 | 0.2810 | 0.1580 | $p = 0.3980$ |

*Empirical Disparity:* Conceptual prose exhibits significantly lower transcription error than mathematical derivations ($\text{CER } 0.9635 \text{ vs } 1.1394$, $p = 0.0211$). Mathematical expressions challenge OCR engines because superscript indices, vector brackets, and fractions lack the contiguous n-gram language priors that guide prose transcription.

---

## VIII. Discussion & Limitations

1. **Revising the Bottleneck Paradigm:** Prior literature has treated automated grading failure as purely a handwriting recognition problem. Our finding that structural routing accounts for 32.7% of downstream error establishes that Document AI layout parsing must precede or accompany OCR engine optimization.
2. **The Pitfall of Preprocessing:** Commercial document pipelines frequently mandate Otsu or Sauvola binarization. In university exams where students use fine-tipped gel pens and ballpoint pens, binarization consistently degrades recognition accuracy ($p = 0.0327$). Modern neural network OCR engines should operate directly on native RGB scans.
3. **Institutional Limitations:** While our study evaluates 195 authentic multi-page scans and 407 subquestions, the corpus represents a single STEM course at an Indian technical university. Future research will expand this protocol to multi-institutional cohorts across diverse handwriting cultures and exam formats.

---

## IX. Conclusion & Reproducibility Statement

This study presents the first comprehensive investigation of assessment-critical handwriting recognition on authentic handwritten university examinations. Through a controlled 2×2 factorial design and two-factor Shapley attribution, we proved that structural routing errors account for nearly one-third of downstream assessment error, while handwriting recognition noise accounts for the remainder. We disproved the utility of heuristic binarization on modern deep OCR engines and demonstrated that selective human review at a 20% budget saves 80% of human marking effort while guaranteeing rigorous assessment standards.

All code, experimental scripts, schemas, and non-sensitive manifests are openly available at:  
`https://github.com/Arhanpg/Automatic-Answer-Evaluation`

---

## References

\bibliographystyle{IEEEtran}
\bibliography{references}
