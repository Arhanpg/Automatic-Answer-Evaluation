"""
scripts/test_decomposition.py -- Stage-wise error decomposition on the locked test partition.
Disentangles:
  Stage 1: Document/Text Recognition
  Stage 2: Structural Answer Routing
  Stage 3: Rubric Interpretation / Semantic Grading
  Stage 4: Assessment-Level Error Attribution
"""
import json
import csv
import pathlib
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, cohen_kappa_score
import jiwer

ROOT = pathlib.Path(__file__).resolve().parent.parent

from backend import criterion_aggregator, exam_schema as es

test_students = ["s7", "s14", "s24", "s29", "s30", "s37"]

# 1. Load Teacher Gold
teacher_gold = {}
for sid in test_students:
    with open(ROOT / f"data/teacher_marks/{sid}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        marks = data["marks"]
        teacher_gold[sid] = {
            "Q1": float(marks.get("1", 0.0)),
            "Q2": float(marks.get("2", 0.0)),
            "Q3": float(marks.get("3", 0.0)),
            "Q4": float(marks.get("4", 0.0)),
            "total": float(sum(marks.values()))
        }

# 2. Load Human Oracle Transcripts
oracle_answers = {}
gt_csv = ROOT / "research_final/03_TRANSCRIPTION_ADJUDICATION.csv"
with open(gt_csv, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sid = row["student_id"]
        if sid in test_students:
            if sid not in oracle_answers:
                oracle_answers[sid] = {}
            oracle_answers[sid][row["subquestion_id"]] = row["student_verbatim_transcript"]

# 3. Load Routed OCR Answers from data/student_answers
routed_answers = {}
for sid in test_students:
    with open(ROOT / f"data/student_answers/{sid}.json", "r", encoding="utf-8") as f:
        routed_answers[sid] = json.load(f)["answers"]

print("=" * 80)
print("STAGE-WISE DECOMPOSITION ON LOCKED TEST SET")
print("=" * 80)

# Evaluate Human Oracle under Frozen Grader
oracle_grades = {}
print("\n[Stage 3: Grader Evaluation on Clean Human Oracle]")
for sid in test_students:
    res = criterion_aggregator.grade_student_answers(sid, oracle_answers[sid])
    oracle_grades[sid] = res["question_scores"]
    print(f"  {sid:3s} | Teacher Gold: {teacher_gold[sid]['total']:4.1f} | Oracle Graded: {oracle_grades[sid]['total']:4.1f} | Delta: {oracle_grades[sid]['total'] - teacher_gold[sid]['total']:+4.1f}")

gold_totals = [teacher_gold[s]["total"] for s in test_students]
oracle_totals = [oracle_grades[s]["total"] for s in test_students]
mae_orc = mean_absolute_error(gold_totals, oracle_totals)
qwk_orc = cohen_kappa_score(np.round(np.array(gold_totals)*2).astype(int), np.round(np.array(oracle_totals)*2).astype(int), weights="quadratic")
print(f"-> Oracle MAE vs Gold: {mae_orc:.3f} marks | QWK: {qwk_orc:.4f}")

# Evaluate End-to-End Routed OCR Answers under Frozen Grader
routed_grades = {}
print("\n[Stage 2 & 1: Grader Evaluation on End-to-End Routed OCR Answers]")
for sid in test_students:
    res = criterion_aggregator.grade_student_answers(sid, routed_answers[sid])
    routed_grades[sid] = res["question_scores"]
    print(f"  {sid:3s} | Teacher Gold: {teacher_gold[sid]['total']:4.1f} | Routed OCR Graded: {routed_grades[sid]['total']:4.1f} | Delta vs Oracle: {routed_grades[sid]['total'] - oracle_grades[sid]['total']:+4.1f}")

routed_totals = [routed_grades[s]["total"] for s in test_students]
mae_routed = mean_absolute_error(gold_totals, routed_totals)
qwk_routed = cohen_kappa_score(np.round(np.array(gold_totals)*2).astype(int), np.round(np.array(routed_totals)*2).astype(int), weights="quadratic") if len(set(routed_totals)) > 1 else 0.0
print(f"-> Routed OCR MAE vs Gold: {mae_routed:.3f} marks | QWK: {qwk_routed:.4f}")

# Routing Analysis: Check which subquestions had answers present in Oracle vs empty in Routed
print("\n[Routing Disentanglement: Question-by-Question Slot Status]")
subqs = ["1a", "1b", "1c", "1d", "2a", "2b", "2c", "3a", "3b", "3c", "4"]
total_items = 0
dropped_items = 0
misrouted_items = 0
intact_items = 0

for sid in test_students:
    for sq in subqs:
        ref_text = oracle_answers[sid].get(sq, "").strip()
        hyp_text = routed_answers[sid].get(sq, "").strip()
        if not ref_text or "omitted" in ref_text.lower():
            continue
        total_items += 1
        if not hyp_text:
            dropped_items += 1
        else:
            # Check if hyp_text has low similarity with ref_text (possibly misrouted text from another question)
            overlap = set(ref_text.lower().split()) & set(hyp_text.lower().split())
            precision = len(overlap) / max(1, len(hyp_text.lower().split()))
            if precision < 0.15 and len(hyp_text.split()) > 10:
                misrouted_items += 1
            else:
                intact_items += 1

print(f"  Total Valid Answer Items: {total_items}")
print(f"  Dropped Items (Empty in slot due to missed header): {dropped_items} ({dropped_items/total_items*100:.1f}%)")
print(f"  Misrouted Items (Text from another question placed in slot): {misrouted_items} ({misrouted_items/total_items*100:.1f}%)")
print(f"  Intact Routed Items: {intact_items} ({intact_items/total_items*100:.1f}%)")
