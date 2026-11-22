"""Summarise evaluation-mode sessions from interaction logs.

Usage (from the repo root):
    python eval/summarise_sessions.py                                        # runtime/interaction_log.csv
    python eval/summarise_sessions.py logs/*.csv                             # several downloaded logs
    python eval/summarise_sessions.py --include-incomplete                   # also unfinished sessions
    python eval/summarise_sessions.py eval/sample_session_log_SIMULATED.csv  # the simulated sample

Prints the tables and writes, to eval/session_summaries/:
    <timestamp>_sessions.csv    one row per session
    <timestamp>_scenarios.csv   per memory-sources setting and demo scenario
    <timestamp>_summary.md      the tables in Markdown

The outcomes are defined in mpa/analysis.py. Descriptive only.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mpa.analysis import (MEASURES, SETTING_NAMES, by_scenario, condition_table,  # noqa: E402
                          load_rows, markdown_table, summarise_conditions, summarise_sessions)

DEFAULT_LOG = ROOT / "runtime" / "interaction_log.csv"
OUT_DIR = ROOT / "eval" / "session_summaries"


def _print_table(rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for j, r in enumerate(rows):
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
        if j == 0:
            print("  ".join("-" * w for w in widths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", help="interaction log CSVs (default: runtime/interaction_log.csv)")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="include sessions that did not finish the questionnaire")
    parser.add_argument("--note", default="", help="e.g. 'simulated sample'")
    parser.add_argument("--out", default=str(OUT_DIR), help="folder for the result files")
    args = parser.parse_args()
    out_dir = Path(args.out)
    # Windows consoles and pipes may use a legacy code page: never crash on a character it lacks
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    paths = [Path(p) for pattern in (args.logs or [str(DEFAULT_LOG)]) for p in glob.glob(pattern)]
    if not paths:
        parser.error("no log files found; download one from 'Demo and evaluation controls' or pass a path")
    rows = load_rows(paths)
    sessions = summarise_sessions(rows)
    if not sessions:
        print("No evaluation sessions found (an evaluation session has at least one prediction).")
        return

    complete = [p for p in sessions if p["complete"]]
    included = sessions if args.include_incomplete else complete
    print(f"Logs: {len(paths)} file(s) · evaluation sessions: {len(sessions)} "
          f"({len(complete)} complete) · summarised: {len(included)}"
          + (f" · {args.note}" if args.note else ""))
    if not included:
        print("No complete sessions yet. Finish a session with the questionnaire, "
              "or rerun with --include-incomplete.")
        return

    table = condition_table(summarise_conditions(included, complete_only=False))
    print("\nBy memory-sources setting (mean ± SD over sessions)\n")
    _print_table(table)

    scen = by_scenario(rows, sessions={p["session"] for p in included})
    scen_table = [["Memory sources", "Scenario", "Tasks", "Prediction exact", "Fit accuracy",
                   "Appropriate reliance"]]
    for s in scen:
        scen_table.append([SETTING_NAMES.get(s["condition"], s["condition"]), s["scenario"], str(s["tasks"])] +
                          ["–" if s[k] is None else f"{s[k]:.2f}"
                           for k in ("prediction_exact", "fit_accuracy", "reliance_appropriate")])
    print("\nBy scenario (pooled over tasks)\n")
    _print_table(scen_table)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["session", "condition", "mode", "complete"] + [k for k, _, _ in MEASURES]
    with (out_dir / f"{stamp}_sessions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sessions)
    with (out_dir / f"{stamp}_scenarios.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scen[0].keys()))
        writer.writeheader()
        writer.writerows(scen)
    md = [f"# Evaluation-mode summary ({stamp})", "",
          f"Sessions summarised: {len(included)} of {len(sessions)} "
          f"({'including' if args.include_incomplete else 'excluding'} incomplete sessions).",
          f"Note: {args.note}" if args.note else "",
          "Descriptive only; higher is better except over-/under-reliance. "
          "Effort is reverse-scored in the questionnaire row.", "",
          "## By memory-sources setting", "", markdown_table(table), "",
          "## By scenario", "", markdown_table(scen_table), ""]
    (out_dir / f"{stamp}_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved to {out_dir}: {stamp}_sessions.csv, _scenarios.csv, _summary.md")


if __name__ == "__main__":
    main()
