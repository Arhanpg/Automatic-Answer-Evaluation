@echo off
cd /d "%~dp0"
set AAE_MODE=demo
if exist "..\..\.venv\Scripts\python.exe" (
    set "PY=..\..\.venv\Scripts\python.exe"
) else if exist "..\.venv\Scripts\python.exe" (
    set "PY=..\.venv\Scripts\python.exe"
) else if exist ".\.venv\Scripts\python.exe" (
    set "PY=.\.venv\Scripts\python.exe"
) else if exist ".\venv\Scripts\python.exe" (
    set "PY=.\venv\Scripts\python.exe"
) else (
    set "PY=python"
)
%PY% app.py
pause
