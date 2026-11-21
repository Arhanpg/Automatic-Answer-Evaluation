# Phase 8 Privacy, PII, and Repository Cleanliness Audit

## 1. Executive Summary
- **Total Tracked Files Audited:** 463
- **Tracked Binary Image/PDF Files:** 43
- **Files Containing Student Roll Numbers:** 2
- **Files Containing Hardcoded Local Windows Paths:** 26
- **Phase 8 Privacy Status:** REMEDIATION_REQUIRED

---

## 2. Institutional Roll Numbers Audit
- `scripts/sanitize_remaining_pii.py`: matches `{'23BCS045'}`
- `tests/test_components.py`: matches `{'21CS7'}`

---

## 3. Local Machine Paths Audit
- `backend/dataset_validation.py`
- `backend/split_validation.py`
- `research_audit/dataset_manifest.csv`
- `research_audit/dataset_visual_audit.csv`
- `research_audit/generate_audit.py`
- `research_audit/phase2_preprocessing_samples.csv`
- `research_audit/run_phase2_audit.py`
- `research_phase3/answer_region_manifest.csv`
- `research_phase3/build_page_question_mapping.py`
- `research_phase3/detailed_ocr_reference_review.csv`
- `research_phase3/page_review_manifest.csv`
- `research_phase3/run_preprocessing_ablation.py`
- `research_phase4/gpu_trocr_runs.csv`
- `research_v2/00_REPOSITORY_AND_DATA_AUDIT.md`
- `research_v2/KNOWN_ISSUES_AND_BLOCKERS.md`
- `research_v2/manifests/source_hashes.csv`
- `research_v2/reports/GATE_0_AUDIT.md`
- `run_full_evaluation.py`
- `scripts/audit_and_inventory.py`
- `scripts/build_phase1_ground_truth.py`
- `scripts/gate0_audit.py`
- `templates/sheets.html`
- `test.py`
- `tools/process_all_37_fresh.py`
- `tools/read_all_37_covers.py`
- `tools/test_pdf_to_images.py`

---

## 4. Tracked Image Binaries
43 image files are tracked in historical commits. For Phase 8:
- All regenerated answer crops are saved locally in `research_v3/crops/` which is added to `.gitignore`.
- Raw full-page scan images reside outside version control in `AAE_DATA_ROOT`.
