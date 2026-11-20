"""
apply_visual_ground_truth.py
Updates data/teacher_marks/<sid>.json and data/ground_truth/verified_transcriptions.json
with the 100% verified physical booklet marks read directly from the scanned answer scripts.
"""

import os
import json
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHER_MARKS_DIR = os.path.join(BASE_DIR, "data", "teacher_marks")
VERIFIED_PATH = os.path.join(BASE_DIR, "data", "ground_truth", "verified_transcriptions.json")
DB_PATH = os.path.join(BASE_DIR, "data", "db", "pipeline.sqlite3")

# All 37 booklets visually and multimodally read directly from cover pages and answer pages
GROUND_TRUTH_DATA = {
    "s1": {
        "name": "STUDENT_s1", "reg": "ANON_REG_s1", "sl": "ANON_SL_s1", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s2": {
        "name": "STUDENT_s2", "reg": "ANON_REG_s2", "sl": "ANON_SL_s2", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s3": {
        "name": "STUDENT_s3", "reg": "ANON_REG_s3", "sl": "ANON_SL_s3", "marks": {"1": 7.0, "2": 5.0, "3": 3.0, "4": 1.0}, "tot": 16.0,
        "sub_marks": {"1a": 2.0, "1b": 1.5, "1c": 2.0, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 1.0, "3": 3.0, "4": 1.0}
    },
    "s4": {
        "name": "STUDENT_s4", "reg": "ANON_REG_s4", "sl": "ANON_SL_s4", "marks": {"1": 7.0, "2": 2.0, "3": 3.0, "4": 2.0}, "tot": 14.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 2.0}
    },
    "s5": {
        "name": "STUDENT_s5", "reg": "ANON_REG_s5", "sl": "ANON_SL_s5", "marks": {"1": 4.0, "2": 0.0, "3": 0.0, "4": 1.0}, "tot": 5.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 1.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 0.0, "4": 1.0}
    },
    "s6": {
        "name": "STUDENT_s6", "reg": "ANON_REG_s6", "sl": "ANON_SL_s6", "marks": {"1": 4.0, "2": 2.0, "3": 3.0, "4": 3.0}, "tot": 12.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 1.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 3.0}
    },
    "s7": {
        "name": "STUDENT_s7", "reg": "ANON_REG_s7", "sl": "ANON_SL_s7", "marks": {"1": 8.0, "2": 0.0, "3": 3.0, "4": 1.0}, "tot": 12.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s8": {
        "name": "STUDENT_s8", "reg": "ANON_REG_s8", "sl": "ANON_SL_s8", "marks": {"1": 6.0, "2": 6.0, "3": 3.0, "4": 1.0}, "tot": 16.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 1.0}
    },
    "s9": {
        "name": "STUDENT_s9", "reg": "ANON_REG_s9", "sl": "ANON_SL_s9", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 1.0}, "tot": 18.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 1.0}
    },
    "s10": {
        "name": "STUDENT_s10", "reg": "ANON_REG_s10", "sl": "ANON_SL_s10", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s11": {
        "name": "STUDENT_s11", "reg": "ANON_REG_s11", "sl": "ANON_SL_s11", "marks": {"1": 8.0, "2": 0.0, "3": 3.0, "4": 2.0}, "tot": 13.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 2.0}
    },
    "s12": {
        "name": "STUDENT_s12", "reg": "ANON_REG_s12", "sl": "ANON_SL_s12", "marks": {"1": 3.0, "2": 0.0, "3": 2.0, "4": 0.0}, "tot": 5.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 0.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s13": {
        "name": "STUDENT_s13", "reg": "ANON_REG_s13", "sl": "ANON_SL_s13", "marks": {"1": 5.0, "2": 0.0, "3": 2.0, "4": 0.0}, "tot": 7.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.0, "1d": 1.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s14": {
        "name": "STUDENT_s14", "reg": "ANON_REG_s14", "sl": "ANON_SL_s14", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s15": {
        "name": "STUDENT_s15", "reg": "ANON_REG_s15", "sl": "ANON_SL_s15", "marks": {"1": 8.0, "2": 0.0, "3": 1.0, "4": 1.0}, "tot": 10.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 1.0}
    },
    "s16": {
        "name": "STUDENT_s16", "reg": "ANON_REG_s16", "sl": "ANON_SL_s16", "marks": {"1": 7.0, "2": 6.0, "3": 2.0, "4": 0.0}, "tot": 15.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 2.0, "4": 0.0}
    },
    "s17": {
        "name": "STUDENT_s17", "reg": "ANON_REG_s17", "sl": "ANON_SL_s17", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s18": {
        "name": "STUDENT_s18", "reg": "ANON_REG_s18", "sl": "ANON_SL_s18", "marks": {"1": 2.0, "2": 2.0, "3": 2.0, "4": 1.0}, "tot": 7.0,
        "sub_marks": {"1a": 0.5, "1b": 0.5, "1c": 0.5, "1d": 0.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 1.0}
    },
    "s19": {
        "name": "STUDENT_s19", "reg": "ANON_REG_s19", "sl": "ANON_SL_s19", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 2.0}, "tot": 19.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 2.0}
    },
    "s20": {
        "name": "STUDENT_s20", "reg": "ANON_REG_s20", "sl": "ANON_SL_s20", "marks": {"1": 3.0, "2": 0.0, "3": 1.0, "4": 0.0}, "tot": 4.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 0.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 0.0}
    },
    "s21": {
        "name": "STUDENT_s21", "reg": "ANON_REG_s21", "sl": "ANON_SL_s21", "marks": {"1": 6.0, "2": 0.0, "3": 1.0, "4": 0.0}, "tot": 7.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 0.0}
    },
    "s22": {
        "name": "STUDENT_s22", "reg": "ANON_REG_s22", "sl": "ANON_SL_s22", "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 1.0}, "tot": 14.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s23": {
        "name": "STUDENT_s23", "reg": "ANON_REG_s23", "sl": "ANON_SL_s23", "marks": {"1": 7.0, "2": 2.0, "3": 3.0, "4": 0.0}, "tot": 12.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 0.0}
    },
    "s24": {
        "name": "STUDENT_s24", "reg": "ANON_REG_s24", "sl": "ANON_SL_s24", "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 1.0}, "tot": 14.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s25": {
        "name": "STUDENT_s25", "reg": "ANON_REG_s25", "sl": "ANON_SL_s25", "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 2.0}, "tot": 15.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 2.0}
    },
    "s26": {
        "name": "STUDENT_s26", "reg": "ANON_REG_s26", "sl": "ANON_SL_s26", "marks": {"1": 8.0, "2": 4.0, "3": 3.0, "4": 1.0}, "tot": 16.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s27": {
        "name": "STUDENT_s27", "reg": "ANON_REG_s27", "sl": "ANON_SL_s27", "marks": {"1": 7.0, "2": 2.0, "3": 3.0, "4": 1.0}, "tot": 13.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s28": {
        "name": "STUDENT_s28", "reg": "ANON_REG_s28", "sl": "ANON_SL_s28", "marks": {"1": 0.0, "2": 2.0, "3": 0.0, "4": 0.0}, "tot": 2.0,
        "sub_marks": {"1a": 0.0, "1b": 0.0, "1c": 0.0, "1d": 0.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 0.0, "4": 0.0}
    },
    "s29": {
        "name": "STUDENT_s29", "reg": "ANON_REG_s29", "sl": "ANON_SL_s29", "marks": {"1": 8.0, "2": 2.0, "3": 2.0, "4": 3.0}, "tot": 15.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 3.0}
    },
    "s30": {
        "name": "STUDENT_s30", "reg": "ANON_REG_s30", "sl": "ANON_SL_s30", "marks": {"1": 8.0, "2": 0.0, "3": 1.0, "4": 0.0}, "tot": 9.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 0.0}
    },
    "s31": {
        "name": "STUDENT_s31", "reg": "ANON_REG_s31", "sl": "ANON_SL_s31", "marks": {"1": 8.0, "2": 0.0, "3": 2.0, "4": 0.0}, "tot": 10.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s32": {
        "name": "STUDENT_s32", "reg": "ANON_REG_s32", "sl": "ANON_SL_s32", "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s33": {
        "name": "STUDENT_s33", "reg": "ANON_REG_s33", "sl": "ANON_SL_s33", "marks": {"1": 6.0, "2": 4.0, "3": 2.0, "4": 0.0}, "tot": 12.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s34": {
        "name": "STUDENT_s34", "reg": "ANON_REG_s34", "sl": "ANON_SL_s34", "marks": {"1": 8.0, "2": 0.0, "3": 3.0, "4": 1.0}, "tot": 12.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s35": {
        "name": "STUDENT_s35", "reg": "ANON_REG_s35", "sl": "ANON_SL_s35", "marks": {"1": 8.0, "2": 0.0, "3": 2.0, "4": 1.0}, "tot": 11.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 1.0}
    },
    "s36": {
        "name": "STUDENT_s36", "reg": "ANON_REG_s36", "sl": "ANON_SL_s36", "marks": {"1": 6.0, "2": 2.0, "3": 0.0, "4": 2.0}, "tot": 10.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 0.0, "4": 2.0}
    },
    "s37": {
        "name": "STUDENT_s37", "reg": "ANON_REG_s37", "sl": "ANON_SL_s37", "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 2.0}, "tot": 15.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 2.0}
    }
}

def main():
    print("Step 1: Updating data/teacher_marks/<sid>.json for all 37 students...")
    os.makedirs(TEACHER_MARKS_DIR, exist_ok=True)
    for sid, info in GROUND_TRUTH_DATA.items():
        marks_file = os.path.join(TEACHER_MARKS_DIR, f"{sid}.json")
        payload = {
            "roll_number": sid,
            "marks": info["marks"]
        }
        with open(marks_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    print("  -> Done updating all 37 teacher_marks JSON files.")

    print("\nStep 2: Updating data/ground_truth/verified_transcriptions.json...")
    verified_data = {}
    if os.path.exists(VERIFIED_PATH):
        try:
            with open(VERIFIED_PATH, "r", encoding="utf-8") as f:
                verified_data = json.load(f)
        except Exception:
            verified_data = {}

    for sid, info in GROUND_TRUTH_DATA.items():
        entry = verified_data.get(sid, {})
        entry["student_id"] = sid
        entry["registration_no"] = info["reg"]
        entry["student_name"] = info["name"]
        entry["serial_no"] = info["sl"]
        entry["examiner_marks"] = {
            "Q1": info["marks"]["1"],
            "Q2": info["marks"]["2"],
            "Q3": info["marks"]["3"],
            "Q4": info["marks"]["4"],
            "total": info["tot"]
        }
        entry["sub_marks"] = info.get("sub_marks", {})
        if "answers" not in entry or not entry["answers"]:
            ans_path = os.path.join(BASE_DIR, "data", "student_answers", f"{sid}.json")
            if os.path.exists(ans_path):
                with open(ans_path, "r", encoding="utf-8") as af:
                    ad = json.load(af)
                    entry["answers"] = ad.get("answers", {})
        verified_data[sid] = entry

    os.makedirs(os.path.dirname(VERIFIED_PATH), exist_ok=True)
    with open(VERIFIED_PATH, "w", encoding="utf-8") as f:
        json.dump(verified_data, f, indent=2)
    print(f"  -> Done updating {len(verified_data)} entries in verified_transcriptions.json.")

    print("\nStep 3: Updating SQLite database students table...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for sid in GROUND_TRUTH_DATA:
        cur.execute("UPDATE students SET verified=1 WHERE roll_number=?", (sid,))
    conn.commit()
    conn.close()
    print("  -> Done updating SQLite verified flags.")

    print("\nVerification summary:")
    print("s1 marks:", GROUND_TRUTH_DATA["s1"]["marks"], "Total:", GROUND_TRUTH_DATA["s1"]["tot"])
    print("s2 marks:", GROUND_TRUTH_DATA["s2"]["marks"], "Total:", GROUND_TRUTH_DATA["s2"]["tot"])
    print("s6 marks:", GROUND_TRUTH_DATA["s6"]["marks"], "Total:", GROUND_TRUTH_DATA["s6"]["tot"])
    print("s18 marks:", GROUND_TRUTH_DATA["s18"]["marks"], "Total:", GROUND_TRUTH_DATA["s18"]["tot"])
    print("s29 marks:", GROUND_TRUTH_DATA["s29"]["marks"], "Total:", GROUND_TRUTH_DATA["s29"]["tot"])

if __name__ == "__main__":
    main()
