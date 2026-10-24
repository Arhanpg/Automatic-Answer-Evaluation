"""
backend/criterion_aggregator.py -- Unified Rubric Criterion Aggregator.

Combines SemanticScorer and MathVerifier to produce transparent, criterion-by-criterion marks
and serializable audit records aligned with the IIIT Dharwad examination schema.
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from backend import semantic_scorer, math_verifier, rubric_builder

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas", "v1", "exam_schema.json")

def load_exam_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def grade_student_answers(student_id: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """
    Grades all questions (Q1..Q4) for a student script.
    Returns:
      {
        "student_id": student_id,
        "question_scores": {"Q1": x, "Q2": y, "Q3": z, "Q4": w, "total": t},
        "sub_scores": {"1a": ..., "1b": ..., "4": ...},
        "criterion_audit": [list of detailed criterion evaluation records]
      }
    """
    schema = load_exam_schema()
    criterion_audit = []
    sub_scores = {}
    question_scores = {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0, "total": 0.0}

    # First, run Math Verifier across Q2, Q3, Q4
    math_results = math_verifier.verify_student_math(answers)

    for q in schema["questions"]:
        qid = q["id"]
        q_awarded = 0.0

        if qid == "1":
            # Pure conceptual question (4 parts x 2 marks = 8 marks)
            for part in q["parts"]:
                pid = part["part_id"]
                ans_text = str(answers.get(pid, answers.get(part["id"], ""))).strip()
                part_awarded = 0.0

                has_contra, contra_reasons = semantic_scorer.detect_contradictions(pid, ans_text)

                if not ans_text:
                    # Omitted sub-question (e.g. s18.pdf skips 1b and 1c)
                    for crit in part["criteria"]:
                        criterion_audit.append({
                            "student_id": student_id,
                            "question_id": pid,
                            "criterion_id": crit["id"],
                            "max_marks": crit["max_marks"],
                            "awarded_marks": 0.0,
                            "student_evidence_quote": "(omitted)",
                            "sympy_equivalence_passed": False,
                            "nli_contradiction_detected": False,
                            "q3c_or_q4_alternative_rule_triggered": False,
                            "justification": "Sub-question omitted by student."
                        })
                    sub_scores[pid] = 0.0
                    continue

                for crit in part["criteria"]:
                    cov_score = semantic_scorer.score_criterion_semantic(ans_text, crit["description"], part["model_answer"])
                    # Discretize to 0.0, 0.5, or 1.0
                    if cov_score >= 0.55:
                        c_marks = crit["max_marks"]
                    elif cov_score >= 0.30:
                        c_marks = round(crit["max_marks"] * 0.5, 1)
                    else:
                        c_marks = 0.0

                    if has_contra:
                        c_marks = 0.0

                    part_awarded += c_marks
                    criterion_audit.append({
                        "student_id": student_id,
                        "question_id": pid,
                        "criterion_id": crit["id"],
                        "max_marks": crit["max_marks"],
                        "awarded_marks": c_marks,
                        "student_evidence_quote": ans_text[:120],
                        "sympy_equivalence_passed": False,
                        "nli_contradiction_detected": has_contra,
                        "q3c_or_q4_alternative_rule_triggered": False,
                        "justification": f"Semantic match score: {cov_score:.2f}." if not has_contra else f"Contradiction: {contra_reasons[0]}"
                    })

                part_awarded = min(part["max_marks"], part_awarded)
                sub_scores[pid] = part_awarded
                q_awarded += part_awarded

        elif qid == "2":
            # Mathematical attention question (6 marks)
            q2_eval = math_results["Q2"]
            q_awarded = q2_eval["marks"]
            details = q2_eval["details"]

            sub_scores["2a"] = details["q2a_marks"]
            sub_scores["2b"] = details["q2b_marks"]
            sub_scores["2c"] = details["q2c_marks"]

            for part in q["parts"]:
                pid = part["part_id"]
                p_key = f"{pid}_marks"
                p_awarded = details.get(p_key, 0.0)
                ans_text = str(answers.get(pid, answers.get(part["id"], ""))).strip()

                for idx, crit in enumerate(part["criteria"]):
                    c_awarded = min(crit["max_marks"], p_awarded / len(part["criteria"]))
                    criterion_audit.append({
                        "student_id": student_id,
                        "question_id": pid,
                        "criterion_id": crit["id"],
                        "max_marks": crit["max_marks"],
                        "awarded_marks": c_awarded,
                        "student_evidence_quote": ans_text[:120] or "(omitted)",
                        "sympy_equivalence_passed": c_awarded == crit["max_marks"],
                        "nli_contradiction_detected": False,
                        "q3c_or_q4_alternative_rule_triggered": False,
                        "justification": f"SymPy vector check: {c_awarded:g}/{crit['max_marks']:g} marks."
                    })

        elif qid == "3":
            # Decoding and sampling question (3 marks)
            q3_eval = math_results["Q3"]
            q_awarded = q3_eval["marks"]
            details = q3_eval["details"]

            sub_scores["3a"] = details["q3a_marks"]
            sub_scores["3b"] = details["q3b_marks"]
            sub_scores["3c"] = details["q3c_marks"]

            for part in q["parts"]:
                pid = part["part_id"]
                p_key = f"{pid}_marks"
                p_awarded = details.get(p_key, 0.0)
                ans_text = str(answers.get(pid, answers.get(part["id"], ""))).strip()
                crit = part["criteria"][0]

                is_contra = (pid == "3b" and p_awarded == 0.0 and bool(ans_text))
                is_override = (pid == "3c" and p_awarded == 1.0)

                criterion_audit.append({
                    "student_id": student_id,
                    "question_id": pid,
                    "criterion_id": crit["id"],
                    "max_marks": crit["max_marks"],
                    "awarded_marks": p_awarded,
                    "student_evidence_quote": ans_text[:120] or "(omitted)",
                    "sympy_equivalence_passed": p_awarded == crit["max_marks"],
                    "nli_contradiction_detected": is_contra,
                    "q3c_or_q4_alternative_rule_triggered": is_override,
                    "justification": f"Decoding evaluation: {p_awarded:g}/{crit['max_marks']:g}."
                })

        elif qid == "4":
            # Laplace Add-1 smoothed bigram (3 marks)
            q4_eval = math_results["Q4"]
            q_awarded = q4_eval["marks"]
            details = q4_eval["details"]
            ans_text = str(answers.get("4", "")).strip()
            sub_scores["4"] = q_awarded

            c_marks_list = [details["q4_c1_marks"], details["q4_c2_marks"], details["q4_c3_marks"]]
            part = q["parts"][0]

            for idx, crit in enumerate(part["criteria"]):
                c_awarded = c_marks_list[idx]
                criterion_audit.append({
                    "student_id": student_id,
                    "question_id": "4",
                    "criterion_id": crit["id"],
                    "max_marks": crit["max_marks"],
                    "awarded_marks": c_awarded,
                    "student_evidence_quote": ans_text[:120] or "(omitted)",
                    "sympy_equivalence_passed": c_awarded == crit["max_marks"],
                    "nli_contradiction_detected": False,
                    "q3c_or_q4_alternative_rule_triggered": (idx == 2 and c_awarded == 1.0),
                    "justification": f"Laplace smoothing criterion {idx+1}: {c_awarded:g}/{crit['max_marks']:g} marks."
                })

        question_scores[f"Q{qid}"] = min(q["max_marks"], q_awarded)

    question_scores["total"] = sum([question_scores[f"Q{i}"] for i in range(1, 5)])

    return {
        "student_id": student_id,
        "question_scores": question_scores,
        "sub_scores": sub_scores,
        "criterion_audit": criterion_audit
    }
