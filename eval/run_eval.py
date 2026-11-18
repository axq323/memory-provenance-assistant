"""Run the fixed test cases against the configured model and score them.

Usage (from the repo root):
    python eval/run_eval.py                    # mode from .env (replay by default)
    MODE=live python eval/run_eval.py --note "v2: added rule about scope"

Prints a per-case table, writes eval/results/<timestamp>_<mode>.csv and
appends a line to eval/prompt_log.md so every prompt version keeps its
score (Lecture 09: change one thing, rerun the same cases, log it).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from mpa.context import PROMPT_VERSION, assemble  # noqa: E402
from mpa.llm import ModelError, ReplayModel, get_model  # noqa: E402
from mpa.memory_store import MemoryStore  # noqa: E402
from mpa.task_types import classify  # noqa: E402
from mpa.validate import validate  # noqa: E402

CASES = ROOT / "eval" / "cases.json"
RESULTS = ROOT / "eval" / "results"
PROMPT_LOG = ROOT / "eval" / "prompt_log.md"


def score_case(case: dict, model, store: MemoryStore) -> dict:
    task, trigger = classify(case["prompt"])
    record = assemble(case["prompt"], store.all(), task, trigger)
    try:
        reply = validate(model.generate(record, case["prompt"]), record.sent_ids)
    except ModelError as exc:
        return {"id": case["id"], "task": task, "used": "", "dropped": "", "C1": False, "C2": False,
                "C3": False, "C4": False, "pass": False, "error": str(exc)}
    used = {u["id"] for u in reply.used}
    checks = {
        "C1": reply.parse_ok,
        "C2": not reply.dropped,
        "C3": set(case["should_use"]) <= used,
        "C4": not (set(case["should_not_use"]) & used),
    }
    return {"id": case["id"], "task": task, "used": ",".join(sorted(used)),
            "dropped": ",".join(d["id"] for d in reply.dropped), **checks,
            "pass": all(checks.values()), "error": ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="what changed since the last run")
    args = parser.parse_args()

    model = get_model()
    mode = "replay" if isinstance(model, ReplayModel) else "live"
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    if mode == "replay":
        cases = [c for c in cases if c.get("replay")]

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(path=Path(tmp) / "memories.json")  # fresh seed copy
        rows = [score_case(c, model, store) for c in cases]

    passed = sum(r["pass"] for r in rows)
    print(f"{'case':5} {'task':10} {'used':10} C1 C2 C3 C4  pass")
    for r in rows:
        marks = " ".join("✓ " if r[k] else "✗ " for k in ("C1", "C2", "C3", "C4"))
        print(f"{r['id']:5} {r['task']:10} {r['used']:10} {marks} {'PASS' if r['pass'] else 'FAIL'}"
              + (f"  ({r['error']})" if r["error"] else ""))
    print(f"\n{passed}/{len(rows)} passed · mode={mode} · model={model.name} · prompt {PROMPT_VERSION}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{stamp}_{mode}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with PROMPT_LOG.open("a", encoding="utf-8") as f:
        f.write(f"| {stamp} | {PROMPT_VERSION} | {mode}: {model.name} | {passed}/{len(rows)} "
                f"| {args.note or '—'} | `{out.relative_to(ROOT)}` |\n")
    print(f"Results: {out.relative_to(ROOT)} · logged in eval/prompt_log.md")


if __name__ == "__main__":
    main()
