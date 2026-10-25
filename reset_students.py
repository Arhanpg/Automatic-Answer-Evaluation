"""
reset_students.py -- Utility script to reset/clear student database records and cached OCR files.

Usage:
    python reset_students.py [--keep-files]
"""
import sys
import argparse
from backend import database as db

def main():
    parser = argparse.ArgumentParser(description="Reset student database and cached files.")
    parser.add_argument("--keep-files", action="store_true", help="Clear only SQLite database records, keep JSON/page files")
    args = parser.parse_args()

    clear_files = not args.keep_files
    print(f"[*] Resetting student database (clear_files={clear_files})...")
    db.reset_students(clear_files=clear_files)
    counts = db.counts()
    print(f"[OK] Reset complete. Current student count in DB: {counts['total']}")

if __name__ == "__main__":
    main()
