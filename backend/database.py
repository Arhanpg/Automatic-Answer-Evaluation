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
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

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

CREATE TABLE IF NOT EXISTS file_cache (
    file_sha256 TEXT PRIMARY KEY,
    file_name TEXT,
    page_count INTEGER,
    metadata_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS ocr_cache (
    cache_key TEXT PRIMARY KEY,
    file_sha256 TEXT,
    student_id TEXT,
    engine_version TEXT,
    ocr_payload_json TEXT,
    cer REAL DEFAULT 0.0,
    confidence REAL DEFAULT 1.0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS background_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT,
    status TEXT DEFAULT 'pending',             -- pending | running | completed | failed | paused
    progress_pct REAL DEFAULT 0.0,
    total_steps INTEGER DEFAULT 0,
    completed_steps INTEGER DEFAULT 0,
    current_item TEXT DEFAULT '',
    current_stage TEXT DEFAULT '',
    vram_mb REAL DEFAULT 0.0,
    logs_json TEXT DEFAULT '[]',
    result_summary_json TEXT DEFAULT '{}',
    error_text TEXT DEFAULT '',
    started_at TEXT,
    updated_at TEXT
);
"""
_COLS = {"source_path", "n_pages", "ocr_status", "split", "verified", "flags"}


@contextlib.contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
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


# =========================================================================
# File & OCR Content-Addressed Caching Helpers
# =========================================================================

def get_file_cache(file_sha256: str):
    with get_conn() as c:
        r = c.execute("SELECT * FROM file_cache WHERE file_sha256=?", (file_sha256,)).fetchone()
        return dict(r) if r else None


def upsert_file_cache(file_sha256: str, file_name: str, page_count: int, metadata: dict = None):
    meta_json = json.dumps(metadata or {})
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with get_conn() as c:
        c.execute(
            """INSERT INTO file_cache (file_sha256, file_name, page_count, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(file_sha256) DO UPDATE SET
                   file_name=excluded.file_name,
                   page_count=excluded.page_count,
                   metadata_json=excluded.metadata_json""",
            (file_sha256, file_name, page_count, meta_json, created_at)
        )


def get_ocr_cache(cache_key: str):
    with get_conn() as c:
        r = c.execute("SELECT * FROM ocr_cache WHERE cache_key=?", (cache_key,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["ocr_payload"] = json.loads(d.get("ocr_payload_json") or "{}")
        return d


def upsert_ocr_cache(cache_key: str, file_sha256: str, student_id: str,
                     engine_version: str, ocr_payload: dict,
                     cer: float = 0.0, confidence: float = 1.0):
    payload_json = json.dumps(ocr_payload)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with get_conn() as c:
        c.execute(
            """INSERT INTO ocr_cache
               (cache_key, file_sha256, student_id, engine_version, ocr_payload_json, cer, confidence, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                   ocr_payload_json=excluded.ocr_payload_json,
                   cer=excluded.cer,
                   confidence=excluded.confidence,
                   created_at=excluded.created_at""",
            (cache_key, file_sha256, student_id, engine_version, payload_json, cer, confidence, created_at)
        )


# =========================================================================
# Resumable Background Jobs Helpers
# =========================================================================

def create_background_job(job_id: str, job_type: str, total_steps: int = 0) -> dict:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with get_conn() as c:
        c.execute(
            """INSERT INTO background_jobs
               (job_id, job_type, status, progress_pct, total_steps, completed_steps,
                current_item, current_stage, vram_mb, logs_json, result_summary_json, started_at, updated_at)
               VALUES (?, ?, 'running', 0.0, ?, 0, '', 'init', 0.0, '[]', '{}', ?, ?)""",
            (job_id, job_type, total_steps, started_at, started_at)
        )
    return get_background_job(job_id)


def update_background_job(job_id: str, **kwargs):
    valid_cols = {
        "status", "progress_pct", "total_steps", "completed_steps",
        "current_item", "current_stage", "vram_mb", "logs_json",
        "result_summary_json", "error_text"
    }
    updates = {k: v for k, v in kwargs.items() if k in valid_cols}
    if not updates:
        return
    if "logs_json" in updates and not isinstance(updates["logs_json"], str):
        updates["logs_json"] = json.dumps(updates["logs_json"])
    if "result_summary_json" in updates and not isinstance(updates["result_summary_json"], str):
        updates["result_summary_json"] = json.dumps(updates["result_summary_json"])
    updates["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with get_conn() as c:
        set_clauses = [f"{col}=?" for col in updates.keys()]
        values = list(updates.values()) + [job_id]
        c.execute(f"UPDATE background_jobs SET {', '.join(set_clauses)} WHERE job_id=?", values)


def get_background_job(job_id: str):
    with get_conn() as c:
        r = c.execute("SELECT * FROM background_jobs WHERE job_id=?", (job_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["logs"] = json.loads(d.get("logs_json") or "[]")
        d["result_summary"] = json.loads(d.get("result_summary_json") or "{}")
        return d


def get_active_background_job(job_type: str = None):
    with get_conn() as c:
        if job_type:
            r = c.execute(
                "SELECT * FROM background_jobs WHERE status='running' AND job_type=? ORDER BY updated_at DESC LIMIT 1",
                (job_type,)
            ).fetchone()
        else:
            r = c.execute(
                "SELECT * FROM background_jobs WHERE status='running' ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if not r:
            # Also check for most recently completed job within last 60 seconds
            r = c.execute(
                "SELECT * FROM background_jobs ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if not r:
            return None
        d = dict(r)
        d["logs"] = json.loads(d.get("logs_json") or "[]")
        d["result_summary"] = json.loads(d.get("result_summary_json") or "{}")
        return d

