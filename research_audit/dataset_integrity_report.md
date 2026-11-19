# Dataset integrity report

| Check | Result | Evidence |
|---|---|---|
| Expected student IDs | PASS: s1-s37 present in source audit | `outputs/dataset_audit/corpus_file_audit.json` |
| Expected response count | PASS: 37 x 4 = 148 in manifest | `outputs/dataset_audit/student_split_manifest.json` |
| Split overlap | PASS: no train/validation/test ID overlap | `backend/split_validation.py` and manifest |
| Source page count | RECORDED: 195 pages | corpus audit JSON |
| Independent label provenance | UNVERIFIED | no annotation log or second grader stored |
| Duplicate/near-duplicate answer audit | NOT ESTABLISHED | no canonical deduplication report found |
| Training JSONL | FAIL for training claim: empty | `data/models/training_set.jsonl` |
| Test independence | LIMITED | only six student groups in locked test |

This report is an evidence inventory, not a claim that the dataset is suitable
for publication without additional validation.
