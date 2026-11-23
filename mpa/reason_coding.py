"""Coding the written reasons: did the person name the scope error, or only notice the mismatch?

The fit question ("did this memory suit your request?") only shows whether someone judged a
misapplied memory unsuitable. "Its tone is too formal" and "the formal preference was for
emailing my manager" both answer No, but only the second shows they understood WHY the memory
didn't fit: it was saved in a different kind of situation (its scope). That understanding is
coded by hand from the reason each person wrote after reacting to the reply (asked before the
fit question, so the fit question can't prime it), using the rubric below.

Codes:
  scope     names where the memory belongs, or the situation it came from (the scope error)
  mismatch  notices the reply is wrong in the memory's direction, but not why
  none      doesn't relate the reply to the memory

The sheet hides the memory-sources setting and shuffles the rows, so the coder is blind to it.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .analysis import parse_detail

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "data" / "memories.seed.json"

CODES = ("scope", "mismatch", "none")

# Per misapplied scenario: what "scope" means here, with an example of each code.
RUBRIC = {
    "S3-birthday": {
        "memory": "m2",
        "scope": "says the formal-tone preference is for work (emails to a manager), not for a friend",
        "examples": {"scope": "The formal preference was for emailing my manager, not for Sam.",
                     "mismatch": "Its tone is too formal.",
                     "none": "I liked it."},
    },
    "S2-explain": {
        "memory": "m1",
        "scope": "says the bullet-point preference is for quick lists such as essay feedback, "
                 "not for an explanation asked for step by step",
        "examples": {"scope": "Short bullets were for my essay feedback, not for explaining things.",
                     "mismatch": "Too short, not enough detail.",
                     "none": "It answered the question."},
    },
}

SHEET_FIELDS = ["item", "session", "trial_id", "scenario", "memory", "what_scope_means", "reason",
                "code", "notes"]


def misapplied_reasons(rows: list[dict]) -> list[dict]:
    """One entry per task whose reply showed a misapplied memory and that has a written reason."""
    seed = {m["id"]: m["text"] for m in json.loads(SEED.read_text(encoding="utf-8"))}
    finished, reasons = {}, {}
    for r in rows:
        key = (r["session_id"], r.get("trial_id", ""))
        if r["event"] == "task_finished":
            finished[key] = parse_detail(r["detail"])
        elif r["event"] == "reason_given":
            reasons[key] = parse_detail(r["detail"]).get("text", "")
    out = []
    for key, d in sorted(finished.items()):
        shown = d.get("misapplied_shown", "none")
        if shown == "none" or key not in reasons or not key[1]:
            continue
        scenario = d.get("scenario", "unknown")
        out.append({"session": key[0], "trial_id": key[1], "scenario": scenario,
                    "memory": "; ".join(f"{m}: {seed.get(m, '?')}" for m in shown.split(",")),
                    "what_scope_means": RUBRIC.get(scenario, {}).get("scope", ""),
                    "reason": reasons[key]})
    return out


def write_sheet(rows: list[dict], out: Path, seed: int = 0) -> int:
    """Write a blind, shuffled coding sheet. Returns the number of reasons to code."""
    items = misapplied_reasons(rows)
    random.Random(seed).shuffle(items)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for i, item in enumerate(items, start=1):
            writer.writerow({"item": i, **item, "code": "", "notes": ""})
    return len(items)


def load_codes(path: Path) -> tuple[dict[tuple[str, str], str], list[str]]:
    """{(session, trial_id): code} from a filled-in sheet, plus problems (blank or unknown codes)."""
    codes, problems = {}, []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row.get("code", "").strip().lower()
            if code in CODES:
                codes[(row["session"], row["trial_id"])] = code
            else:
                problems.append(f"item {row.get('item')}: " + (f"unknown code {code!r}" if code else "not coded"))
    return codes, problems


def rubric_markdown() -> str:
    lines = ["# Rubric for coding the written reasons", "",
             "Code each reason with one of: `scope`, `mismatch`, `none`.", "",
             "- **scope**: names where the memory belongs, or the situation it came from (the scope error)",
             "- **mismatch**: notices the reply is wrong in the memory's direction, but not why",
             "- **none**: doesn't relate the reply to the memory", "",
             "The sheet hides the memory-sources setting and shuffles the rows. Code every row before "
             "looking at any results. If possible, have a second person code them too and compare.", ""]
    for sid, r in RUBRIC.items():
        lines += [f"## {sid}", "", f"`scope` here means the reason {r['scope']}.", ""]
        lines += [f"- {code}: “{example}”" for code, example in r["examples"].items()] + [""]
    return "\n".join(lines)
