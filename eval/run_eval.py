"""Run the fixed test cases against the configured model and score them.

Usage (from the repo root):
    python eval/run_eval.py                                   # mode and prompt from .env
    python eval/run_eval.py --repeat 3 --note "v2 baseline"   # 3 runs, per-case summary
    python eval/run_eval.py --repeat 3 --prompt v3            # compare a prompt version
    python eval/run_eval.py --repeat 3 --condition corrected  # with the user's corrections applied
    python eval/run_eval.py --cases E09 E11                   # only some cases

Prints a per-case table (one run) or a per-case summary (several runs),
writes eval/results/<timestamp>_<mode>.csv and appends a line to
eval/prompt_log.md so every prompt version keeps its score (Lecture 09:
change one thing, rerun the same cases, log it).

Conditions:
  default    the seed memories, nothing restricted
  corrected  the corrections a user made in test D: the bullet-point (m1) and
             formal-tone (m2) memories are not used for personal messages.
             Measures whether scoped correction fixes the over-application cases.

Note: the criteria score what the model REPORTS using. A memory can shape a
reply without being reported; only the influence check in the app measures that.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from mpa.context import PROMPT_VERSION, SYSTEM_PROMPTS, assemble  # noqa: E402
from mpa.llm import ModelError, ReplayModel, get_model  # noqa: E402
from mpa.memory_store import MemoryStore  # noqa: E402
from mpa.task_types import classify  # noqa: E402
from mpa.validate import validate  # noqa: E402

CASES = ROOT / "eval" / "cases.json"
RESULTS = ROOT / "eval" / "results"
PROMPT_LOG = ROOT / "eval" / "prompt_log.md"

CONDITIONS = {
    "default": [],
    "corrected": [("m1", "personal_message"), ("m2", "personal_message")],
}


def score_case(case: dict, model, store: MemoryStore, prompt_version: str | None = None) -> dict:
    task, trigger = classify(case["prompt"])
    record = assemble(case["prompt"], store.all(), task, trigger, prompt_version=prompt_version)
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


def run_once(cases: list[dict], model, condition: str, prompt_version: str) -> list[dict]:
    """One run over the cases with a fresh copy of the seed memories."""
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(path=Path(tmp) / "memories.json")
        for memory_id, task in CONDITIONS[condition]:
            store.exclude_for_task(memory_id, task)
        return [score_case(c, model, store, prompt_version) for c in cases]


def summarise_runs(runs: list[list[dict]]) -> list[dict]:
    """Per case: how often it passed, and how often each memory was reported."""
    summary = []
    for rows in zip(*runs):
        reported = Counter(m for r in rows for m in r["used"].split(",") if m)
        summary.append({
            "id": rows[0]["id"], "task": rows[0]["task"],
            "passed": sum(r["pass"] for r in rows), "runs": len(rows),
            "reported": ", ".join(f"{m} {n}/{len(rows)}" for m, n in sorted(reported.items())) or "-",
            "failed": ",".join(sorted({k for r in rows for k in ("C1", "C2", "C3", "C4") if not r[k]})) or "-",
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="what changed since the last run")
    parser.add_argument("--repeat", type=int, default=1, help="runs over all cases (replies vary)")
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default="default")
    parser.add_argument("--prompt", choices=sorted(SYSTEM_PROMPTS), default=PROMPT_VERSION,
                        help=f"system prompt version (default {PROMPT_VERSION})")
    parser.add_argument("--cases", nargs="+", help="only these case ids, e.g. E09 E11")
    args = parser.parse_args()

    model = get_model()
    mode = "replay" if isinstance(model, ReplayModel) else "live"
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    if mode == "replay":
        cases = [c for c in cases if c.get("replay")]
    if args.cases:
        cases = [c for c in cases if c["id"] in set(args.cases)]
        if not cases:
            parser.error("none of those case ids exist (in this mode)")

    runs = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"run {i + 1}/{args.repeat} …", flush=True)
        runs.append(run_once(cases, model, args.condition, args.prompt))

    scores = [sum(r["pass"] for r in rows) for rows in runs]
    n = len(cases)
    if args.repeat == 1:
        print(f"{'case':5} {'task':16} {'used':10} C1 C2 C3 C4  pass")
        for r in runs[0]:
            marks = " ".join("✓ " if r[k] else "✗ " for k in ("C1", "C2", "C3", "C4"))
            print(f"{r['id']:5} {r['task']:16} {r['used']:10} {marks} {'PASS' if r['pass'] else 'FAIL'}"
                  + (f"  ({r['error']})" if r["error"] else ""))
    else:
        print(f"\n{'case':5} {'task':16} {'passed':7} {'failed':12} reported (runs)")
        for s in summarise_runs(runs):
            print(f"{s['id']:5} {s['task']:16} {s['passed']}/{s['runs']:<5} {s['failed']:12} {s['reported']}")
    score_text = (f"{scores[0]}/{n}" if args.repeat == 1 else
                  f"mean {sum(scores) / len(scores):.1f}/{n} (runs: {', '.join(map(str, scores))})")
    print(f"\n{score_text} · mode={mode} · model={model.name} · prompt {args.prompt} "
          f"· condition={args.condition}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{stamp}_{mode}_{args.prompt}_{args.condition}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        fields = ["run", "prompt", "condition"] + list(runs[0][0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, rows in enumerate(runs, start=1):
            writer.writerows({"run": i, "prompt": args.prompt, "condition": args.condition, **r} for r in rows)
    note = (f"[{args.condition}] " if args.condition != "default" else "") + (args.note or "—")
    with PROMPT_LOG.open("a", encoding="utf-8") as f:
        f.write(f"| {stamp} | {args.prompt} | {mode}: {model.name} | {score_text} "
                f"| {note} | `{out.relative_to(ROOT)}` |\n")
    print(f"Results: {out.relative_to(ROOT)} · logged in eval/prompt_log.md")


if __name__ == "__main__":
    main()
