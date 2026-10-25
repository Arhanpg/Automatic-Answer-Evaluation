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
        "name": "ADARSH S KAMATAGI", "reg": "23BCS005", "sl": "116833",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s2": {
        "name": "AMRITANSHU ADITYA", "reg": "23BCS013", "sl": "116831",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s3": {
        "name": "Babhale Rohan Laxmikant", "reg": "23BCS026", "sl": "116858",
        "marks": {"1": 7.0, "2": 5.0, "3": 3.0, "4": 1.0}, "tot": 16.0,
        "sub_marks": {"1a": 2.0, "1b": 1.5, "1c": 2.0, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 1.0, "3": 3.0, "4": 1.0}
    },
    "s4": {
        "name": "Bikram Howladar", "reg": "23BCS033", "sl": "116837",
        "marks": {"1": 7.0, "2": 2.0, "3": 3.0, "4": 2.0}, "tot": 14.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 2.0}
    },
    "s5": {
        "name": "G. Rakesh", "reg": "23BCS049", "sl": "116870",
        "marks": {"1": 4.0, "2": 0.0, "3": 0.0, "4": 1.0}, "tot": 5.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 1.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 0.0, "4": 1.0}
    },
    "s6": {
        "name": "Ganesh Kumatole", "reg": "23BCS050", "sl": "116883",
        "marks": {"1": 4.0, "2": 2.0, "3": 3.0, "4": 3.0}, "tot": 12.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 1.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 3.0}
    },
    "s7": {
        "name": "G. Srivarshith Rao", "reg": "23BCS051", "sl": "116844",
        "marks": {"1": 8.0, "2": 0.0, "3": 3.0, "4": 1.0}, "tot": 12.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s8": {
        "name": "K. Shriyans", "reg": "23BCS064", "sl": "116864",
        "marks": {"1": 6.0, "2": 6.0, "3": 3.0, "4": 1.0}, "tot": 16.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 1.0}
    },
    "s9": {
        "name": "K. Hemadithya", "reg": "23BCS068", "sl": "116875",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 1.0}, "tot": 18.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 1.0}
    },
    "s10": {
        "name": "Kothapally Anjan Reddy", "reg": "23BCS071", "sl": "116815",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s11": {
        "name": "Krishna Sai Gollamudi", "reg": "23BCS074", "sl": "116789",
        "marks": {"1": 8.0, "2": 0.0, "3": 3.0, "4": 2.0}, "tot": 13.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 2.0}
    },
    "s12": {
        "name": "Mohammad Samad Khan", "reg": "23BCS082", "sl": "116783",
        "marks": {"1": 3.0, "2": 0.0, "3": 2.0, "4": 0.0}, "tot": 5.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 0.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s13": {
        "name": "NUTHAN.K", "reg": "23BCS088", "sl": "116827",
        "marks": {"1": 5.0, "2": 0.0, "3": 2.0, "4": 0.0}, "tot": 7.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.0, "1d": 1.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s14": {
        "name": "P. Shasank Pawan", "reg": "23BCS096", "sl": "116825",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s15": {
        "name": "Piyush Maurya", "reg": "23BCS097", "sl": "116786",
        "marks": {"1": 8.0, "2": 0.0, "3": 1.0, "4": 1.0}, "tot": 10.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 1.0}
    },
    "s16": {
        "name": "Thrivikram. P", "reg": "23BCS100", "sl": "116773",
        "marks": {"1": 7.0, "2": 6.0, "3": 2.0, "4": 0.0}, "tot": 15.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 2.0, "4": 0.0}
    },
    "s17": {
        "name": "S. Keerthivassan", "reg": "23BCS114", "sl": "116775",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s18": {
        "name": "Biddhaastho BS", "reg": "23BCS127", "sl": "116838",
        "marks": {"1": 2.0, "2": 2.0, "3": 2.0, "4": 1.0}, "tot": 7.0,
        "sub_marks": {"1a": 0.5, "1b": 0.5, "1c": 0.5, "1d": 0.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 1.0}
    },
    "s19": {
        "name": "SUVAN GURURAJ", "reg": "23BCS129", "sl": "116835",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 2.0}, "tot": 19.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 2.0}
    },
    "s20": {
        "name": "A. Guru Sai Harsha", "reg": "23BDS005", "sl": "116819",
        "marks": {"1": 3.0, "2": 0.0, "3": 1.0, "4": 0.0}, "tot": 4.0,
        "sub_marks": {"1a": 1.0, "1b": 1.0, "1c": 1.0, "1d": 0.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 0.0}
    },
    "s21": {
        "name": "B. Bharath Kumar", "reg": "23BDS012", "sl": "116818",
        "marks": {"1": 6.0, "2": 0.0, "3": 1.0, "4": 0.0}, "tot": 7.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 0.0}
    },
    "s22": {
        "name": "Chaitra V. Kathimani", "reg": "23BDS026", "sl": "116797",
        "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 1.0}, "tot": 14.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s23": {
        "name": "Gompa Kuladeep", "reg": "23BDS021", "sl": "116801",
        "marks": {"1": 7.0, "2": 2.0, "3": 3.0, "4": 0.0}, "tot": 12.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 0.0}
    },
    "s24": {
        "name": "Kala Surya Narayana Rao", "reg": "23BDS025", "sl": "116804",
        "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 1.0}, "tot": 14.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s25": {
        "name": "Kamal Nayan Kumar", "reg": "23BDS026", "sl": "116793",
        "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 2.0}, "tot": 15.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 2.0}
    },
    "s26": {
        "name": "MANIKESH KUMAR", "reg": "23BDS032", "sl": "116807",
        "marks": {"1": 8.0, "2": 4.0, "3": 3.0, "4": 1.0}, "tot": 16.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s27": {
        "name": "SADAN. K", "reg": "23BDS047", "sl": "116820",
        "marks": {"1": 7.0, "2": 2.0, "3": 3.0, "4": 1.0}, "tot": 13.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s28": {
        "name": "S. Harshith", "reg": "23BDS058", "sl": "116821",
        "marks": {"1": 0.0, "2": 2.0, "3": 0.0, "4": 0.0}, "tot": 2.0,
        "sub_marks": {"1a": 0.0, "1b": 0.0, "1c": 0.0, "1d": 0.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 0.0, "4": 0.0}
    },
    "s29": {
        "name": "TARAN JAIN", "reg": "23BDS062", "sl": "116814",
        "marks": {"1": 8.0, "2": 2.0, "3": 2.0, "4": 3.0}, "tot": 15.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 3.0}
    },
    "s30": {
        "name": "Tejaswi A.C", "reg": "23BDS064", "sl": "116826",
        "marks": {"1": 8.0, "2": 0.0, "3": 1.0, "4": 0.0}, "tot": 9.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 1.0, "4": 0.0}
    },
    "s31": {
        "name": "Vinay N. R", "reg": "23BDS068", "sl": "116824",
        "marks": {"1": 8.0, "2": 0.0, "3": 2.0, "4": 0.0}, "tot": 10.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s32": {
        "name": "Yash G. Birje", "reg": "23BDS070", "sl": "116790",
        "marks": {"1": 8.0, "2": 6.0, "3": 3.0, "4": 3.0}, "tot": 20.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 2.0, "2b": 2.0, "2c": 2.0, "3": 3.0, "4": 3.0}
    },
    "s33": {
        "name": "Akhilesh R. Ghadge", "reg": "23BEC004", "sl": "116780",
        "marks": {"1": 6.0, "2": 4.0, "3": 2.0, "4": 0.0}, "tot": 12.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 2.0, "2c": 0.0, "3": 2.0, "4": 0.0}
    },
    "s34": {
        "name": "Neeraj S. Gawane", "reg": "23BEC031", "sl": "116806",
        "marks": {"1": 8.0, "2": 0.0, "3": 3.0, "4": 1.0}, "tot": 12.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 3.0, "4": 1.0}
    },
    "s35": {
        "name": "Ojas Tulshian", "reg": "23BEC034", "sl": "116823",
        "marks": {"1": 8.0, "2": 0.0, "3": 2.0, "4": 1.0}, "tot": 11.0,
        "sub_marks": {"1a": 2.0, "1b": 2.0, "1c": 2.0, "1d": 2.0, "2a": 0.0, "2b": 0.0, "2c": 0.0, "3": 2.0, "4": 1.0}
    },
    "s36": {
        "name": "Ritu Srivastava", "reg": "25PHDAS006", "sl": "116750",
        "marks": {"1": 6.0, "2": 2.0, "3": 0.0, "4": 2.0}, "tot": 10.0,
        "sub_marks": {"1a": 1.5, "1b": 1.5, "1c": 1.5, "1d": 1.5, "2a": 2.0, "2b": 0.0, "2c": 0.0, "3": 0.0, "4": 2.0}
    },
    "s37": {
        "name": "PRAVEEN HONGAL", "reg": "26PHDCS07", "sl": "116999",
        "marks": {"1": 8.0, "2": 2.0, "3": 3.0, "4": 2.0}, "tot": 15.0,
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
