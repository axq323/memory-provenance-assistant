"""Evaluation-mode summary: turn the interaction log into the outcomes from Assignment 1, Section 04.

Each evaluation session (session_id) is summarised first, then sessions are
grouped by the memory-sources setting (A = sources shown, B = sources hidden),
so a session with more tasks does not count more than one with fewer.

  predictive understanding  prediction_made  -> exact-match rate, mean overlap (Jaccard)
  applicability judgement   fit_judged       -> accuracy against ground truth, "not sure" rate;
                                                judged a misapplied memory unsuitable (the fit
                                                answer only: "too formal" counts, so it does
                                                not show the person knew WHY)
  scope diagnosis           reason_given     -> named the scope error, from the written reasons
                            + coded sheet       coded by hand (mpa/reason_coding.py); only when a
                                                coded sheet is passed in
  correction                task_finished    -> on replies with a misapplied memory: restricted
                                                for that kind of request and kept ("scoped", the
                                                appropriate fix) vs forgotten (over-correction).
                                                Kept apart from diagnosis: one can be right
                                                without the other
  behavioural reliance      task_finished    -> appropriate-reliance rate; over-reliance on
                                                misapplied replies (the risk that source context
                                                lends them credibility, Assignment 1 Gap);
                                                under-reliance on suitable replies;
                                                "no decision" (finished without acting) is
                                                reported on its own and left out of those rates
  checking effort           task_finished    -> sources opened and seconds per task
  self-reported attitudes   questionnaire    -> mean per item, kept apart from behaviour
                                                (Bucinca et al., 2021); effort reverse-scored

Only sessions with at least one prediction are evaluation sessions; other log
rows (development and live testing) are ignored. Sessions without a finished
questionnaire are kept but marked incomplete.

Descriptive statistics only. Running the evaluation with users is planned work.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .evaluation import QUESTIONNAIRE

QUESTION_ITEMS = [item for item, _, _ in QUESTIONNAIRE]
REVERSED = {item for item, _, rev in QUESTIONNAIRE if rev}

# Per-session measures, in display order: (key, label, higher is better?)
MEASURES = [
    ("tasks", "Tasks finished", None),
    ("prediction_exact", "Prediction: exact match rate", True),
    ("prediction_overlap", "Prediction: mean overlap (0-1)", True),
    ("fit_accuracy", "Fit judgement: accuracy", True),
    ("fit_unsure", "Fit judgement: 'not sure' rate", None),
    ("misapplied_judged_unsuitable", "Diagnosis: judged the misapplied memory unsuitable (fit answer)", True),
    ("scope_named", "Diagnosis: named the scope error (coded reasons)", True),
    ("correction_scoped", "Correction: scoped and kept (appropriate)", True),
    ("correction_forgot", "Correction: forgotten instead (over-correction)", False),
    ("reliance_appropriate", "Reliance: appropriate rate", True),
    ("over_reliance", "Over-reliance on misapplied replies", False),
    ("under_reliance", "Under-reliance on suitable replies", False),
    ("no_decision", "Reliance: no decision rate", None),
    ("sources_per_task", "Effort: sources opened per task", None),
    ("seconds_per_task", "Effort: seconds per task", None),
] + [(f"q_{i}", f"Questionnaire: {i}" + (" (reversed)" if i in REVERSED else ""), True)
     for i in QUESTION_ITEMS]


# ------------------------------------------------------------------ input ---
def parse_detail(detail: str) -> dict:
    """'a=1;b=x,y' -> {'a': '1', 'b': 'x,y'}. Values may contain '='."""
    out = {}
    for part in (detail or "").split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def load_rows(paths: list[Path]) -> list[dict]:
    """Rows from one or more interaction logs (e.g. one downloaded per session).
    Identical rows are kept once, so the same log can be passed twice safely."""
    seen, rows = set(), []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = tuple(sorted(row.items()))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    rows.sort(key=lambda r: (r["session_id"], r["timestamp"]))   # stable: keeps order within a second
    return rows


def by_session(rows: list[dict]) -> dict[str, list[dict]]:
    sessions = defaultdict(list)
    for r in rows:
        sessions[r["session_id"]].append(r)
    return dict(sessions)


# ----------------------------------------------------------- per session ---
def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def summarise_session(rows: list[dict], codes: dict | None = None) -> dict | None:
    """Outcomes for one session, or None if it has no evaluation tasks.

    codes: {(session_id, trial_id): "scope" | "mismatch" | "none"} from a coded reasons sheet.
    Without it, scope_named is None (not measured)."""
    predictions = [parse_detail(r["detail"]) for r in rows if r["event"] == "prediction_made"]
    if not predictions:
        return None
    fits = [parse_detail(r["detail"]) for r in rows if r["event"] == "fit_judged"]
    tasks = [parse_detail(r["detail"]) for r in rows if r["event"] == "task_finished"]
    answers = {}
    for r in rows:
        if r["event"] == "questionnaire_answered":
            d = parse_detail(r["detail"])
            if d.get("item") in QUESTION_ITEMS:
                answers[d["item"]] = int(d["score"])        # a later answer replaces an earlier one
    conditions = {r["condition"] for r in rows if r["event"] in ("prediction_made", "task_finished")}

    scored_fits = [f for f in fits if f.get("correct") in ("True", "False")]
    misapplied_fits = [f for f in fits if f.get("truth") == "misapplied"]
    with_misapplied = [t for t in tasks if t.get("misapplied_shown", "none") != "none" and "correction" in t]
    misapplied_trials = {r.get("trial_id", "") for r in rows if r["event"] == "task_finished"
                         and parse_detail(r["detail"]).get("misapplied_shown", "none") != "none"}
    coded = [codes[(rows[0]["session_id"], t)] for t in sorted(misapplied_trials)
             if codes and (rows[0]["session_id"], t) in codes]
    annotated = [t for t in tasks if t.get("reliance") not in (None, "n/a")]
    judged = [t for t in annotated if t["reliance"] != "no decision"]
    misapplied = [t for t in judged if t.get("misapplied_shown", "none") != "none"]
    suitable = [t for t in judged if t.get("misapplied_shown", "none") == "none"]

    s = {
        "session": rows[0]["session_id"],
        "condition": conditions.pop() if len(conditions) == 1 else "mixed",
        "mode": rows[0]["mode"],
        "complete": any(r["event"] == "session_finished" for r in rows),
        "tasks": len(tasks),
        "prediction_exact": _rate([p.get("exact") == "True" for p in predictions]),
        "prediction_overlap": _avg([float(p["jaccard"]) for p in predictions if "jaccard" in p]),
        "fit_accuracy": _rate([f["correct"] == "True" for f in scored_fits]),
        "fit_unsure": _rate([f.get("answer") == "unsure" for f in fits]),
        "misapplied_judged_unsuitable": _rate([f.get("answer") == "no" for f in misapplied_fits]),
        "scope_named": _rate([c == "scope" for c in coded]),
        "correction_scoped": _rate([t["correction"].split("+")[0] == "scoped" for t in with_misapplied]),
        "correction_forgot": _rate([t["correction"].split("+")[0] == "forgot" for t in with_misapplied]),
        "reliance_appropriate": _rate([t["reliance"] == "appropriate" for t in judged]),
        "over_reliance": _rate([t["reliance"] == "over-reliance" for t in misapplied]),
        "under_reliance": _rate([t["reliance"] == "under-reliance" for t in suitable]),
        "no_decision": _rate([t["reliance"] == "no decision" for t in annotated]),
        "sources_per_task": _avg([int(t.get("sources_opened", 0)) for t in tasks]),
        "seconds_per_task": _avg([int(t.get("seconds", 0)) for t in tasks]),
    }
    for item in QUESTION_ITEMS:
        score = answers.get(item)
        s[f"q_{item}"] = (8 - score if item in REVERSED else score) if score is not None else None
    return s


def summarise_sessions(rows: list[dict], codes: dict | None = None) -> list[dict]:
    out = [summarise_session(r, codes) for r in by_session(rows).values()]
    return [s for s in out if s is not None]


# ---------------------------------------------------------- per condition ---
def summarise_conditions(sessions: list[dict], complete_only: bool = True) -> dict:
    """{setting: {"n": int, measure: {"mean", "sd", "n"}}} over sessions."""
    groups = defaultdict(list)
    for p in sessions:
        if p["complete"] or not complete_only:
            groups[p["condition"]].append(p)
    out = {}
    for cond, ps in sorted(groups.items()):
        out[cond] = {"n": len(ps)}
        for key, _, _ in MEASURES:
            values = [p[key] for p in ps if p[key] is not None]
            out[cond][key] = {"mean": round(mean(values), 3) if values else None,
                              "sd": round(stdev(values), 3) if len(values) > 1 else None,
                              "n": len(values)}
    return out


def by_scenario(rows: list[dict], sessions: set[str] | None = None) -> list[dict]:
    """Per memory-sources setting and demo scenario, pooled over tasks."""
    cells = defaultdict(lambda: {"tasks": 0, "exact": [], "fit": [], "appropriate": []})
    for r in rows:
        if sessions is not None and r["session_id"] not in sessions:
            continue
        d = parse_detail(r["detail"])
        if r["event"] == "prediction_made":
            cells[(r["condition"], d.get("scenario", "unknown"))]["exact"].append(d.get("exact") == "True")
        elif r["event"] == "fit_judged" and d.get("correct") in ("True", "False"):
            cells[(r["condition"], d.get("scenario", "unknown"))]["fit"].append(d["correct"] == "True")
        elif r["event"] == "task_finished":
            cell = cells[(r["condition"], d.get("scenario", "unknown"))]
            cell["tasks"] += 1
            if d.get("reliance") not in (None, "n/a", "no decision"):
                cell["appropriate"].append(d["reliance"] == "appropriate")
    return [{"condition": c, "scenario": sc, "tasks": v["tasks"],
             "prediction_exact": _rate(v["exact"]), "fit_accuracy": _rate(v["fit"]),
             "reliance_appropriate": _rate(v["appropriate"])}
            for (c, sc), v in sorted(cells.items())]


# ----------------------------------------------------------------- output ---
def _fmt(cell: dict | None) -> str:
    if not cell or cell["mean"] is None:
        return "–"
    return f"{cell['mean']:.2f}" + (f" ± {cell['sd']:.2f}" if cell["sd"] is not None else "") + f" (n={cell['n']})"


SETTING_NAMES = {"A": "Sources shown", "B": "Sources hidden"}


def condition_table(summary: dict) -> list[list[str]]:
    """Rows for a measure x memory-sources-setting table, header first."""
    conds = list(summary)
    header = ["Measure"] + [f"{SETTING_NAMES.get(c, c)} (N={summary[c]['n']})" for c in conds]
    if set(conds) >= {"A", "B"}:
        header.append("Shown - hidden")
    rows = [header]
    for key, label, _ in MEASURES:
        row = [label] + [_fmt(summary[c][key]) for c in conds]
        if set(conds) >= {"A", "B"}:
            a, b = summary["A"][key]["mean"], summary["B"][key]["mean"]
            row.append(f"{a - b:+.2f}" if a is not None and b is not None else "–")
        rows.append(row)
    return rows


def markdown_table(rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
