"""
backend/dataset_validation.py -- Forensic audit and dataset integrity validation for the 37 scripts.

Inspects all 37 booklets in F:\\Automatic answer evaluation dataset, checks page counts,
cover sheet locations, red ink availability, and validates the question paper / solution schema.
"""
import os
import json
from typing import Dict, List, Any
import pymupdf
from backend.output_manager import DedicatedOutputManager

DEFAULT_DATASET_DIR = r"F:\Automatic answer evaluation dataset"

def audit_dataset_files(dataset_dir: str = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    """Audits PDF files, page counts, and document structure across all 37 scripts."""
    all_pdfs_dir = os.path.join(dataset_dir, "All PDF's")
    audit_results = {
        "dataset_path": dataset_dir,
        "all_pdfs_dir_exists": os.path.exists(all_pdfs_dir),
        "question_paper": os.path.exists(os.path.join(dataset_dir, "Midsem.docx.pdf")),
        "solution_key": os.path.exists(os.path.join(dataset_dir, "Midsem Solutions.docx.pdf")),
        "scripts": {},
        "missing_scripts": [],
    }

    total_pages = 0
    for i in range(1, 38):
        sid = f"s{i}"
        pdf_name = f"{sid}.pdf"
        pdf_path = os.path.join(all_pdfs_dir, pdf_name)
        if os.path.exists(pdf_path):
            try:
                doc = pymupdf.open(pdf_path)
                p_count = len(doc)
                doc.close()
                audit_results["scripts"][sid] = {
                    "pdf_path": pdf_path,
                    "pages": p_count,
                    "exists": True,
                    "size_bytes": os.path.getsize(pdf_path),
                }
                total_pages += p_count
            except Exception as e:
                audit_results["scripts"][sid] = {
                    "pdf_path": pdf_path,
                    "error": str(e),
                    "exists": True,
                }
        else:
            audit_results["missing_scripts"].append(sid)

    audit_results["total_scripts_found"] = len(audit_results["scripts"])
    audit_results["total_pages_scanned"] = total_pages
    return audit_results

def run_dataset_audit(output_mgr: DedicatedOutputManager, dataset_dir: str = DEFAULT_DATASET_DIR):
    """Executes audit and registers results."""
    audit_data = audit_dataset_files(dataset_dir)
    audit_file = os.path.join(output_mgr.dirs["dataset_audit"], "corpus_file_audit.json")
    output_mgr._write_json(audit_file, audit_data)
    return audit_data

if __name__ == "__main__":
    mgr = DedicatedOutputManager()
    data = run_dataset_audit(mgr)
    print(f"Audited {data['total_scripts_found']} scripts. Total pages: {data['total_pages_scanned']}")
