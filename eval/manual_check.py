"""Check a benchmark run by hand: does each reply actually do the job?

The automated criteria (eval/run_eval.py, C1-C4) score what the model REPORTS
using. This script turns a results CSV into a sheet for reading the replies
themselves, then scores the filled-in sheet against the automated result.

Usage (from the repo root):
    python eval/manual_check.py make eval/results/<run>.csv            # first run of each case
    python eval/manual_check.py make eval/results/<run>.csv --all      # every run
    python eval/manual_check.py score eval/manual_checks/<sheet>.csv   # after filling it in

Fill in, for each row, with y / n (or na where there is nothing to check):
    answers_request    the reply does what was asked (a full message, an explanation, a recipe ...)
    follows_fitting    it follows the memories in should_use (na if should_use is empty)
    avoids_misapplied  it does NOT follow the memories in should_not_use
    notes              optional

A reply passes the manual check if answers_request is y and the other two are y or na.
`score` prints the manual pass rate, the agreement with the automated pass/fail,
and the disagreements, and appends a summary row to eval/prompt_log.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "eval" / "cases.json"
OUT_DIR = ROOT / "eval" / "manual_checks"
PROMPT_LOG = ROOT / "eval" / "prompt_log.md"

SHEET_FIELDS = ["run", "case", "prompt", "should_use", "should_not_use", "reported_used",
                "automated_pass", "answer", "answers_request", "follows_fitting", "avoids_misapplied",
                "notes"]
CHECKS = ("answers_request", "follows_fitting", "avoids_misapplied")


def make_sheet(results: Path, out_dir: Path = OUT_DIR, every_run: bool = False) -> Path:
    cases = {c["id"]: c for c in json.loads(CASES.read_text(encoding="utf-8"))["cases"]}
    with results.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "answer" not in rows[0]:
        raise SystemExit(f"{results.name} has no 'answer' column. Results saved before 23 Nov 2026 "
                         "don't keep the replies: rerun eval/run_eval.py to get a run you can check.")
    if not every_run:
        rows = [r for r in rows if str(r.get("run", "1")) == "1"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{results.stem}_sheet.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for r in rows:
            c = cases.get(r["id"], {})
            writer.writerow({"run": r.get("run", "1"), "case": r["id"], "prompt": c.get("prompt", ""),
                             "should_use": ",".join(c.get("should_use", [])) or "-",
                             "should_not_use": ",".join(c.get("should_not_use", [])) or "-",
                             "reported_used": r.get("used") or "-", "automated_pass": r["pass"],
                             "answer": r["answer"], "answers_request": "", "follows_fitting": "",
                             "avoids_misapplied": "", "notes": ""})
    return out


def _passes(row: dict) -> bool | None:
    values = [row.get(k, "").strip().lower() for k in CHECKS]
    if not all(values) or any(v not in ("y", "n", "na") for v in values):
        return None                                        # not filled in (properly)
    return values[0] == "y" and all(v in ("y", "na") for v in values[1:])


def score_sheet(sheet: Path) -> dict:
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    checked = [(r, _passes(r)) for r in rows]
    done = [(r, ok) for r, ok in checked if ok is not None]
    auto = lambda r: str(r["automated_pass"]).strip().lower() == "true"   # noqa: E731
    return {
        "rows": len(rows), "checked": len(done),
        "manual_pass": sum(ok for _, ok in done),
        "agree": sum(ok == auto(r) for r, ok in done),
        "disagreements": [(r["run"], r["case"], auto(r), ok) for r, ok in done if ok != auto(r)],
        "unfilled": [f"{r['case']} (run {r['run']})" for r, ok in checked if ok is None],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("make", help="make a sheet from a results CSV")
    m.add_argument("results")
    m.add_argument("--all", action="store_true", help="every run, not just the first")
    m.add_argument("--out", default=str(OUT_DIR))
    s = sub.add_parser("score", help="score a filled-in sheet")
    s.add_argument("sheet")
    s.add_argument("--no-log", action="store_true", help="don't add a row to eval/prompt_log.md")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.command == "make":
        out = make_sheet(Path(args.results), Path(args.out), args.all)
        print(f"Sheet: {out}\nFill in answers_request, follows_fitting and avoids_misapplied "
              "(y / n / na) for each row, then run: python eval/manual_check.py score " + str(out))
        return

    sheet = Path(args.sheet)
    r = score_sheet(sheet)
    if not r["checked"]:
        raise SystemExit("No rows are filled in yet (y / n / na in all three check columns).")
    print(f"Checked {r['checked']} of {r['rows']} replies · manual pass {r['manual_pass']}/{r['checked']} "
          f"· agrees with the automated pass/fail on {r['agree']}/{r['checked']}")
    for run, case, auto_ok, manual_ok in r["disagreements"]:
        print(f"  {case} run {run}: automated {'PASS' if auto_ok else 'FAIL'}, "
              f"manual {'PASS' if manual_ok else 'FAIL'}")
    if r["unfilled"]:
        print("Not filled in: " + ", ".join(r["unfilled"]))
    if not args.no_log and PROMPT_LOG.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with PROMPT_LOG.open("a", encoding="utf-8") as f:
            f.write(f"| `{sheet.name}` ({stamp}) | {sheet.stem.removesuffix('_sheet')} | {r['checked']} "
                    f"| {r['manual_pass']}/{r['checked']} | {r['agree']}/{r['checked']} |\n")
        print("Logged in eval/prompt_log.md")


if __name__ == "__main__":
    main()
