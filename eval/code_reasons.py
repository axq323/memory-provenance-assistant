"""Hand-code the written reasons for the 'named the scope error' measure.

Usage (from the repo root):
    python eval/code_reasons.py make                              # every session in runtime/sessions/
    python eval/code_reasons.py make logs/*.csv --out eval/reason_codes/pilot_sheet.csv
    python eval/code_reasons.py check eval/reason_codes/<sheet>.csv
    python eval/summarise_sessions.py --codes eval/reason_codes/<sheet>.csv

`make` writes a shuffled sheet with one row per task where the reply showed a misapplied
memory, without the memory-sources setting, plus eval/reason_rubric.md. Fill in `code`
(scope / mismatch / none) for every row using the rubric, then `check` it and pass it to
summarise_sessions.py with --codes. See mpa/reason_coding.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mpa.analysis import load_rows  # noqa: E402
from mpa.reason_coding import load_codes, rubric_markdown, write_sheet  # noqa: E402

RUNTIME = Path(os.getenv("MPA_RUNTIME_DIR", ROOT / "runtime"))
DEFAULT_LOGS = [str(RUNTIME / "sessions" / "*" / "interaction_log.csv")]
OUT_DIR = ROOT / "eval" / "reason_codes"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make", help="make a blind coding sheet from interaction logs")
    m.add_argument("logs", nargs="*")
    m.add_argument("--out", default="")
    c = sub.add_parser("check", help="check a filled-in sheet")
    c.add_argument("sheet")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.command == "make":
        paths = [Path(p) for pattern in (args.logs or DEFAULT_LOGS) for p in glob.glob(pattern)]
        if not paths:
            parser.error("no log files found")
        out = Path(args.out) if args.out else OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_reasons_sheet.csv"
        n = write_sheet(load_rows(paths), out)
        rubric = ROOT / "eval" / "reason_rubric.md"
        rubric.write_text(rubric_markdown(), encoding="utf-8")
        print(f"Sheet: {out} ({n} reasons to code)\nRubric: {rubric}\n"
              "Fill in `code` with scope / mismatch / none, then run: "
              f"python eval/code_reasons.py check {out}")
        return

    codes, problems = load_codes(Path(args.sheet))
    counts = {c: list(codes.values()).count(c) for c in ("scope", "mismatch", "none")}
    print(f"Coded {len(codes)} reasons: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for p in problems:
        print("  " + p)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
