# Privacy, Ethical Governance, and Sensitive Student Data Audit

**Auditors**: Arhan Ghosarwade & Abhinav Gumgol  
**Affiliation**: Department of Computer Science & Engineering, IIIT Dharwad  
**Date**: November 22, 2026  
**Audited Baseline**: Branch `research/paper2-model-comparison`

---

## 1. Governance Principles & Compliance

In accordance with institutional research standards, data protection regulations, and academic ethics:
1. **Zero External API Transmission**: No raw scanned images, student handwriting crops, extracted transcripts, or teacher marks may be transmitted over the internet or sent to external closed-source commercial APIs (e.g. OpenAI, Anthropic, Google Cloud Vertex). All model inference in Paper 2 is executed strictly on local offline infrastructure (NVIDIA GeForce RTX 3050 Laptop GPU and local Ollama daemon).
2. **Pseudonymization of Student Records**: All 37 examination booklets are identified exclusively by pseudonymous identifiers (`s1` through `s37`). Roll numbers, student names, registration IDs, course section details, and demographic markers have been stripped from all machine-readable files.
3. **Repository Quarantine of Raw Scans**: Physical high-resolution page scans (`*.jpg`, `*.pdf`, `IMG_*.jpg`) and extracted answer crops (`research_v3/crops/*.png`) are quarantined in local private storage and explicitly excluded from git tracking via `.gitignore`.
4. **Single-Marker Reference Transparency**: Teacher marks represent authentic single-marker institutional evaluations awarded during course grading. We do not claim dual independent human marking or inter-rater reliability for the grade targets.

---

## 2. Verification of Local Artifact Quarantine

| Asset Family | Disk Location | Git Status | Privacy Compliance |
|:---|:---|:---|:---|
| **Raw Scanned Pages** | `data/pages/` | Ignored by `.gitignore` | Passed (quarantined locally) |
| **Handwritten Crops** | `research_v3/crops/` | Ignored by `.gitignore` | Passed (quarantined locally) |
| **Local Annotation Store** | `research_v3/phase8_ground_truth/annotations_store/` | Ignored by `.gitignore` | Passed (quarantined locally) |
| **Pseudonymized Transcripts** | `data/ground_truth/verified_transcriptions.json` | Tracked in Git | Passed (anonymized text only) |
| **Pseudonymized Marks** | `data/teacher_marks/s*.json` | Tracked in Git | Passed (anonymized numeric scores) |
| **Exam Rubrics** | `schemas/v1/exam_schema.json` | Tracked in Git | Passed (public course syllabus) |

---

## 3. Mandatory Model Invocation Controls

- Any prompted LLM evaluations must communicate exclusively with `http://localhost:11434` (Ollama) or local HuggingFace weights.
- Any network call to an external remote endpoint is strictly forbidden and must fail automated regression tests.
