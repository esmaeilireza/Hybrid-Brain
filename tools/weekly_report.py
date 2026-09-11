"""Weekly report - data/logs/<year>-W<NN>.md (generated, not hand-written).

Every number comes from a live command (git, pytest) at generation
time. One manual line at the end: next week's opening item.

Usage: python tools/weekly_report.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "logs"


def sh(*args: str) -> str:
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                       timeout=300)
    return (r.stdout + r.stderr).strip()


def main() -> None:
    today = date.today()
    iso = today.isocalendar()
    monday = today - timedelta(days=today.weekday())
    path = LOGS / f"{iso[0]}-W{iso[1]:02d}.md"

    commits = sh("git", "log", f"--since={monday.isoformat()}",
                 "--pretty=format:- %h %s (%ad)", "--date=short")
    tags = sh("git", "tag", "--sort=-creatordate").splitlines()
    dirty = sh("git", "status", "--porcelain")

    pytest_out = sh(sys.executable, "-m", "pytest", "tests/", "-q")
    m = re.search(r"^(\d+ passed.*)$", pytest_out, re.M)
    test_line = m.group(1).strip() if m else "PYTEST: no summary (broken?)"

    bench = ROOT / "docs" / "benchmarks.md"
    tail = ("\n".join(bench.read_text(encoding="utf-8").splitlines()[-15:])
            if bench.exists() else "(docs/benchmarks.md missing)")

    fence = "`" * 3
    report = (
        f"# Weekly Report - {iso[0]} W{iso[1]:02d} ({today})\n\n"
        f"## Tests (live)\n**{test_line}**\n\n"
        f"## Commits this week\n{commits or '(none)'}\n\n"
        f"## Latest tags\n" + "\n".join(f"- {t}" for t in tags[:5])
        + f"\n\n## Benchmarks (tail)\n{fence}\n{tail}\n{fence}\n\n"
        f"## Uncommitted\n{dirty or '(clean)'}\n\n"
        f"## Next week's opening item\nTO FILL: one roadmap line.\n")

    LOGS.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"written: {path}")
    print(f"tests: {test_line}")


if __name__ == "__main__":
    main()
