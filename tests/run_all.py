"""Run every test module in its own process (they use different data folders / modes)."""
import glob, os, subprocess, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bad = 0
for f in sorted(glob.glob(os.path.join(root, "tests", "test_*.py"))):
    name = "tests." + os.path.basename(f)[:-3]
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "unittest", name], cwd=root, capture_output=True, text=True)
    tail = [l for l in r.stderr.strip().splitlines() if l.startswith(("Ran", "OK", "FAILED"))]
    print(f"{name:32s}", " | ".join(tail) or "no output")
    if r.returncode:
        bad += 1; print(r.stderr[-2500:])
sys.exit(1 if bad else 0)
