"""Show latest errors from app.log. Usage: python scripts/show_errors.py [N]"""
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
log_file = Path("data/logs/app.log")

if not log_file.exists():
    print(f"Log file not found: {log_file}")
    sys.exit(1)

lines = log_file.read_text(encoding="utf-8").splitlines()
errors = [l for l in lines if "ERROR" in l or "CRITICAL" in l or "Traceback" in l]

print(f"Last {min(N, len(errors))} errors from {log_file} ({len(lines)} total lines):\n")
for line in errors[-N:]:
    print(line)
