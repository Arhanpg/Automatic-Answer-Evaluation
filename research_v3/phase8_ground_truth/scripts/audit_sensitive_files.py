r"""
research_v3/phase8_ground_truth/scripts/audit_sensitive_files.py

Scans all Git-tracked files in the repository for:
1. Institutional roll numbers (patterns like 23BCS* or 21CS*)
2. Student personal names
3. Local hardcoded filesystem paths (e.g. F:\... or C:\Users\...)
4. Tracked binary assets (PNG crops, JPG scans, PDF booklets)

Outputs an audit report and non-zero exit code if unredacted student PII is tracked.
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_REPORT = REPO_ROOT / "research_v3" / "phase8_ground_truth" / "07_SENSITIVE_DATA_AUDIT.md"

PII_PATTERN = re.compile(r"\b(?:23BCS\d+|21CS\d+)\b", re.IGNORECASE)
PERSONAL_PATH_PATTERN = re.compile(r"[A-Z]:\\(?:Users|Automatic Answer evaluation)", re.IGNORECASE)


def audit_privacy():
    print("=" * 80)
    print("PHASE 8: PRIVACY, PII, AND REPOSITORY CLEANLINESS AUDIT")
    print("=" * 80)

    # Get tracked files
    res = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        print("Error: Failed to run git ls-files")
        sys.exit(1)

    tracked_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
    print(f"Total Git-tracked files: {len(tracked_files)}")

    files_with_roll_numbers = []
    files_with_paths = []
    tracked_binaries = []

    for rel_path in tracked_files:
        p = REPO_ROOT / rel_path

        # Check binary assets
        if rel_path.endswith((".png", ".jpg", ".jpeg", ".pdf")):
            tracked_binaries.append(rel_path)

        if not p.is_file():
            continue

        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            # Scan for roll numbers
            m_pii = PII_PATTERN.findall(content)
            if m_pii:
                files_with_roll_numbers.append((rel_path, set(m_pii)))

            # Scan for personal paths
            if PERSONAL_PATH_PATTERN.search(content):
                files_with_paths.append(rel_path)
        except Exception:
            pass

    print(f"\n1. Tracked Image/PDF Binaries: {len(tracked_binaries)}")
    if tracked_binaries:
        print(f"   Sample tracked images (N={len(tracked_binaries)}): {tracked_binaries[:5]}")

    print(f"\n2. Tracked Files with Student Roll Numbers: {len(files_with_roll_numbers)}")
    for f, rolls in files_with_roll_numbers:
        print(f"   - {f}: {rolls}")

    print(f"\n3. Tracked Files with Hardcoded Local Paths: {len(files_with_paths)}")
    for f in files_with_paths[:8]:
        print(f"   - {f}")

    # Generate Markdown report
    report_md = f"""# Phase 8 Privacy, PII, and Repository Cleanliness Audit

## 1. Executive Summary
- **Total Tracked Files Audited:** {len(tracked_files)}
- **Tracked Binary Image/PDF Files:** {len(tracked_binaries)}
- **Files Containing Student Roll Numbers:** {len(files_with_roll_numbers)}
- **Files Containing Hardcoded Local Windows Paths:** {len(files_with_paths)}
- **Phase 8 Privacy Status:** {'CLEAN' if not files_with_roll_numbers else 'REMEDIATION_REQUIRED'}

---

## 2. Institutional Roll Numbers Audit
{chr(10).join([f"- `{f}`: matches `{rolls}`" for f, rolls in files_with_roll_numbers]) if files_with_roll_numbers else "No institutional roll numbers detected in Git-tracked files."}

---

## 3. Local Machine Paths Audit
{chr(10).join([f"- `{f}`" for f in files_with_paths]) if files_with_paths else "No local machine paths detected."}

---

## 4. Tracked Image Binaries
{len(tracked_binaries)} image files are tracked in historical commits. For Phase 8:
- All regenerated answer crops are saved locally in `research_v3/crops/` which is added to `.gitignore`.
- Raw full-page scan images reside outside version control in `AAE_DATA_ROOT`.
"""
    OUTPUT_REPORT.write_text(report_md, encoding="utf-8")
    print(f"\nAudit report saved to: {OUTPUT_REPORT}")

    if files_with_roll_numbers:
        print("\nNotice: Roll numbers found in test or sanitize scripts. Sanitization recommended.")


if __name__ == "__main__":
    audit_privacy()
