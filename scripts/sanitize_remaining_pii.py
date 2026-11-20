"""
scripts/sanitize_remaining_pii.py -- Complete PII sanitization.
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

targets = [
    ROOT / "scratch/compare_marks.py",
    ROOT / "tools/apply_visual_ground_truth.py",
]

for p in targets:
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    # Replace individual names, reg numbers and serials
    # Pattern: 'name': '...', 'reg': '23BCS...', 'sl': '...'
    def sub_single(m):
        sid = m.group(1)
        rest = m.group(2)
        return f"'{sid}': {{'name': 'STUDENT_{sid}', 'reg': 'ANON_REG_{sid}', 'sl': 'ANON_SL_{sid}', {rest}"

    text = re.sub(r"'(s\d+)':\s*\{\s*'name':\s*'[^']+',\s*'reg':\s*'[^']+',\s*'sl':\s*'[^']+',\s*([^\}]+)", sub_single, text)

    def sub_double(m):
        sid = m.group(1)
        rest = m.group(2)
        return f'"{sid}": {{\n        "name": "STUDENT_{sid}", "reg": "ANON_REG_{sid}", "sl": "ANON_SL_{sid}", {rest}'

    text = re.sub(r'"(s\d+)":\s*\{\s*\n\s*"name":\s*"[^"]+",\s*"reg":\s*"[^"]+",\s*"sl":\s*"[^"]+",\s*([^\}]+)', sub_double, text)

    p.write_text(text, encoding="utf-8")
    print(f"Sanitized: {p}")

# Also replace in tests/test_sheet_parser.py dummy 23BCS045 with ANON_TEST_REG
test_file = ROOT / "tests/test_sheet_parser.py"
t_text = test_file.read_text(encoding="utf-8")
t_text = t_text.replace("23BCS045", "ANON_TEST_REG")
test_file.write_text(t_text, encoding="utf-8")
print(f"Sanitized: {test_file}")
