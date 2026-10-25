"""
backend/database.py -- pipeline STATE only (who is where in the workflow).

The real content lives in JSON files:
  data/student_answers/<id>.json   FOLDER 1  (answers only)
  data/teacher_marks/<id>.json     FOLDER 2  (marks only)
"""
import contextlib
import json
import sqlite3
import time

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    roll_number TEXT PRIMARY KEY,
    source_path TEXT,
    n_pages     INTEGER DEFAULT 0,
    ocr_status  TEXT DEFAULT 'ocr_done',      -- ocr_done | ocr_failed
    split       TEXT DEFAULT 'unassigned',    -- unassigned | train | test
    verified    INTEGER DEFAULT 0,            -- 1 once a human checked answers+marks
    flags       TEXT DEFAULT '[]',            -- JSON list of QC warnings
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS pipeline_state (key TEXT PRIMARY KEY, value TEXT);
"""
_COLS = {"source_path", "n_pages", "ocr_status", "split", "verified", "flags"}


@contextlib.contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(SCHEMA)
        try:
            existing = {row[1] for row in c.execute("PRAGMA table_info(students)").fetchall()}
            col_defs = {
                "source_path": "TEXT",
                "n_pages": "INTEGER DEFAULT 0",
                "ocr_status": "TEXT DEFAULT 'ocr_done'",
                "split": "TEXT DEFAULT 'unassigned'",
                "verified": "INTEGER DEFAULT 0",
                "flags": "TEXT DEFAULT '[]'",
                "updated_at": "REAL",
            }
            for col, col_type in col_defs.items():
                if col not in existing:
                    c.execute(f"ALTER TABLE students ADD COLUMN {col} {col_type}")
        except Exception:
            pass


def _row(r):
    d = dict(r)
    d["flags"] = json.loads(d.get("flags") or "[]")
    return d


def upsert_student(roll, **fields):
    fields = {k: v for k, v in fields.items() if k in _COLS}
    if "flags" in fields and not isinstance(fields["flags"], str):
        fields["flags"] = json.dumps(fields["flags"])
    with get_conn() as c:
        if c.execute("SELECT 1 FROM students WHERE roll_number=?", (roll,)).fetchone():
            if fields:
                sets = ", ".join(f"{k}=?" for k in fields)
                c.execute(f"UPDATE students SET {sets}, updated_at=? WHERE roll_number=?",
                          (*fields.values(), time.time(), roll))
        else:
            cols = ["roll_number", "updated_at", *fields]
            c.execute(f"INSERT INTO students ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      (roll, time.time(), *fields.values()))


def get_student(roll):
    with get_conn() as c:
        r = c.execute("SELECT * FROM students WHERE roll_number=?", (roll,)).fetchone()
        return _row(r) if r else None


def list_students(split=None):
    with get_conn() as c:
        if split:
            rows = c.execute("SELECT * FROM students WHERE split=?", (split,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM students").fetchall()
    out = [_row(r) for r in rows]
    out.sort(key=lambda s: _natural(s["roll_number"]))
    return out


def _natural(s):
    import re
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def counts():
    studs = list_students()
    return {
        "total": len(studs),
        "ocr_failed": sum(1 for s in studs if s["ocr_status"] == "ocr_failed"),
        "train": sum(1 for s in studs if s["split"] == "train"),
        "test": sum(1 for s in studs if s["split"] == "test"),
        "verified": sum(1 for s in studs if s["verified"]),
        "train_verified": sum(1 for s in studs if s["split"] == "train" and s["verified"]),
        "flagged": sum(1 for s in studs if s["flags"]),
    }


def get_state(key, default=None):
    with get_conn() as c:
        r = c.execute("SELECT value FROM pipeline_state WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default


def set_state(key, value):
    with get_conn() as c:
        c.execute("INSERT INTO pipeline_state (key,value) VALUES (?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def reset_students(clear_files=True):
    import os, shutil
    with get_conn() as c:
        c.execute("DELETE FROM students")
        c.execute("DELETE FROM pipeline_state WHERE key = 'selection'")
    if clear_files:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        for sub in ["student_answers", "teacher_marks", "pages", "ocr_output"]:
            folder = os.path.join(data_dir, sub)
            if os.path.exists(folder):
                for item in os.listdir(folder):
                    if item == ".gitkeep":
                        continue
                    p = os.path.join(folder, item)
                    try:
                        if os.path.isdir(p):
                            shutil.rmtree(p)
                        else:
                            os.remove(p)
                    except Exception:
                        pass
