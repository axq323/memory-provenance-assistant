"""Evaluation mode: ground truth and scoring for the guided tasks.

Evaluation mode lets the design be tested in the way Assignment 1 (Section 04)
proposes. Each part maps to one outcome:

  predictive understanding  -> score_prediction(): predicted vs actually used memories
  applicability judgement   -> score_fit(): "did this memory suit your request?" vs ground truth
  behavioural reliance      -> summarise_task() + reliance(): what the user did with
                               the reply (accept, flag, correct), judged against ground truth
  self-reported attitudes   -> QUESTIONNAIRE at the end of the session, logged apart from
                               behaviour because the two can diverge (Bucinca et al., 2021)

The fit question is asked AFTER the user has reacted to the reply,
so asking it cannot prime the behaviour that measures reliance. Before it,
a neutral question ("What made you choose that?") is asked, so whether the
user saw the scope error can also be coded from their own words.

Two outcomes are kept apart for misapplied memories (they can diverge):
  diagnosis   -> the fit judgement: did the user say the memory didn't suit the request?
  correction  -> correction_outcome(): did they fix it the right way (restrict it for
                 this kind of request), or over-correct (forget it), or undo it?

Ground truth comes from the annotated replay scenarios (data/replay.json):
which memories fit each demo request and which would be misapplied. It is
looked up by prompt, so it also applies when the same prompts are asked in
live mode.
"""

from __future__ import annotations

import json
import os

from .llm import REPLAY_PATH, ReplayModel

_scenarios = ReplayModel()


def _truth(s: dict) -> dict:
    return {"scenario": s["id"], "fits": list(s.get("fits", [])),
            "misapplied": list(s.get("misapplied", []))}


def ground_truth(user_text: str) -> dict | None:
    """{"scenario", "fits", "misapplied"} for a demo prompt, or None if unknown."""
    s = _scenarios.match(user_text)
    return None if s is None else _truth(s)


def scenario(scenario_id: str) -> dict | None:
    return next((s for s in _scenarios.scenarios if s["id"] == scenario_id), None)


def truth_for(scenario_id: str | None) -> dict | None:
    """Ground truth for a scheduled task, looked up by its scenario id (not by prompt text)."""
    s = scenario(scenario_id) if scenario_id else None
    return None if s is None else _truth(s)


def carries_from(scenario_id: str) -> str | None:
    """The task whose end state a follow-up task keeps (data/replay.json, evaluation_carry_over).
    Every other task starts from the seeded memories."""
    data = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    return (data.get("evaluation_carry_over") or {}).get(scenario_id)


def evaluation_schedule() -> list[str]:
    """The scenario ids evaluation mode serves, in order (data/replay.json, evaluation_schedule).
    MPA_EVAL_SCHEDULE="S3-birthday,S1-dinner" overrides it; unknown ids are an error."""
    override = os.getenv("MPA_EVAL_SCHEDULE", "").strip()
    if override:
        ids = [i.strip() for i in override.split(",") if i.strip()]
    else:
        data = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
        ids = data.get("evaluation_schedule") or [s["id"] for s in data["scenarios"]]
    unknown = [i for i in ids if scenario(i) is None]
    if unknown:
        raise ValueError(f"Unknown scenario ids in the evaluation schedule: {unknown}")
    return ids


def score_prediction(predicted: set[str], used: set[str]) -> dict:
    """Exact match and Jaccard overlap between predicted and actually used memories.

    Both empty counts as a perfect prediction ("none of them" and none were used).
    """
    union = predicted | used
    jaccard = 1.0 if not union else len(predicted & used) / len(union)
    return {"exact": predicted == used, "jaccard": round(jaccard, 2)}


FIT_ANSWERS = {"yes": "Yes, it suited my request", "no": "No, it didn't suit my request",
               "unsure": "Not sure"}


def score_fit(answer: str, memory_id: str, truth: dict | None) -> dict:
    """Score one applicability judgement against ground truth.

    truth label: "fits", "misapplied" or "unknown" (not annotated, e.g. a live prompt).
    correct: True/False, or None when the truth is unknown. "Not sure" counts as
    not correct, and is flagged separately so the analysis can report it.
    """
    if truth and memory_id in truth["fits"]:
        label = "fits"
    elif truth and memory_id in truth["misapplied"]:
        label = "misapplied"
    else:
        label = "unknown"
    if label == "unknown":
        correct = None
    else:
        correct = (answer == "yes" and label == "fits") or (answer == "no" and label == "misapplied")
    return {"truth": label, "correct": correct, "unsure": answer == "unsure"}


CORRECTION_EVENTS = {"not_for_task", "memory_edited", "memory_forgotten"}


def summarise_task(rows: list[dict], started_at: str, finished_at: str) -> dict:
    """What the user did between seeing the reply and finishing the task.

    rows: interaction-log rows that belong to this task only (same trial_id: its
    reply, rewrites of that reply, and memory-panel actions while it was open).
    Returns counts, the memory ids they corrected, and the decision they made:
    "accepted", "flagged", "corrected" (joined with "+" if several) or "none".
    """
    from datetime import datetime
    events = [r["event"] for r in rows]
    corrected = sorted({r["memory_id"] for r in rows if r["event"] in CORRECTION_EVENTS and r["memory_id"]})
    seconds = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
    accepted, flagged = "reply_accepted" in events, "reply_flagged" in events
    decision = "+".join(name for name, did in (("accepted", accepted), ("flagged", flagged),
                                              ("corrected", bool(corrected))) if did) or "none"
    return {"accepted": accepted, "flagged": flagged, "corrected": corrected, "decision": decision,
            "sources_opened": events.count("source_opened"), "seconds": int(seconds)}


def reliance(misapplied_shown: set[str], fitting_shown: set[str], behaviour: dict) -> str:
    """Was the user's reaction appropriate (Assignment 1: accept suitable
    personalisation, question misapplied memories)?

    Only the reply's own memories count as a decision about it: accepting or flagging the
    reply, or correcting a memory the reply used. Correcting some other memory says nothing
    about whether they relied on this reply.

    no decision at all             -> "no decision" (finished without accepting, flagging
        or correcting anything). Kept apart, so doing nothing is never scored as a choice.
    only another memory corrected  -> "unrelated correction" (no accept or flag, and no
        correction of a memory in the reply). Kept apart too: it is neither trusting nor
        questioning the reply.
    misapplied memory in the reply -> "appropriate" if they flagged the reply or
        corrected that memory, otherwise "over-reliance" (they accepted it).
    only fitting memories          -> "under-reliance" if they flagged the reply or
        corrected a fitting memory, otherwise "appropriate" (accepted it).
    no annotated memory            -> "n/a".
    """
    corrected = set(behaviour["corrected"])
    in_reply = misapplied_shown | fitting_shown
    if not in_reply:
        return "n/a"
    relevant = corrected & in_reply
    if not (behaviour.get("accepted") or behaviour.get("flagged") or relevant):
        return "unrelated correction" if corrected else "no decision"
    if misapplied_shown:
        questioned = behaviour["flagged"] or bool(relevant & misapplied_shown)
        return "appropriate" if questioned else "over-reliance"
    questioned = behaviour["flagged"] or bool(relevant & fitting_shown)
    return "under-reliance" if questioned else "appropriate"


def correction_actions(rows: list[dict]) -> list[str]:
    """The corrections in one task, in order, with their scope:
    restrict:m2@personal_message, undo:m2@personal_message, edit:m2, forget:m2."""
    out = []
    for r in rows:
        mid, task = r.get("memory_id", ""), r.get("task", "")
        if r["event"] == "not_for_task":
            out.append(f"restrict:{mid}@{task}")
        elif r["event"] == "undo_not_for_task":
            out.append(f"undo:{mid}@{task}")
        elif r["event"] == "memory_edited":
            out.append(f"edit:{mid}")
        elif r["event"] == "memory_forgotten":
            out.append(f"forget:{mid}")
    return out


def final_state(start: dict, now: dict | None, task: str) -> str:
    """Where one memory ended up when the task finished, after any undo.

    start: the memory as it was when the task began ({"text", "excluded_tasks"}).
    now:   the memory in the store at the end, or None if it was forgotten.
    Returns "forgotten", or "unchanged", or a "+"-joined mix of
    "restricted" (no longer used for this task's type), "restricted elsewhere"
    (newly restricted for a different type only) and "edited".
    """
    if now is None:
        return "forgotten"
    parts = []
    added = set(now.get("excluded_tasks", [])) - set(start.get("excluded_tasks", []))
    if task in added:
        parts.append("restricted")
    elif added:
        parts.append("restricted elsewhere")
    if now.get("text") != start.get("text"):
        parts.append("edited")
    return "+".join(parts) or "unchanged"


def correction_outcome(misapplied: set[str], fitting: set[str], states: dict[str, str],
                       actions: list[str]) -> str:
    """Was the correction the right one? Scored apart from whether the user questioned
    the reply (reliance) and from whether they judged the memory's fit (diagnosis).

    For a misapplied memory the appropriate correction keeps the memory but stops it
    applying to this kind of request:
      "scoped"       restricted for this task type and still stored (appropriate)
      "forgot"       deleted, so it is lost where it does fit (over-correction)
      "edited"       reworded but not restricted: may or may not fix the scope (check by hand)
      "wrong scope"  restricted, but only for a different kind of request
      "undone"       corrected, then undone, so nothing changed
      "none"         not corrected
    A fitting memory that was changed adds "+unneeded". With only fitting memories:
    "unneeded" if any was changed, otherwise "none". Nothing annotated: "n/a".
    """
    changed_fitting = any(states.get(m, "unchanged") != "unchanged" for m in fitting)
    if not misapplied:
        if not fitting:
            return "n/a"
        return "unneeded" if changed_fitting else "none"
    outcomes = []
    for m in sorted(misapplied):
        state = states.get(m, "unchanged")
        if state == "forgotten":
            outcomes.append("forgot")
        elif state.split("+")[0] == "restricted":
            outcomes.append("scoped")
        elif "edited" in state:
            outcomes.append("edited")
        elif "restricted elsewhere" in state:
            outcomes.append("wrong scope")
        elif any(a.split(":", 1)[1].split("@")[0] == m for a in actions):
            outcomes.append("undone")
        else:
            outcomes.append("none")
    return ",".join(outcomes) + ("+unneeded" if changed_fitting else "")


# End-of-session questionnaire (7-point agreement). Items map to Assignment 1:
# self-reported trust, perceived understanding and control, checking effort (a
# usability cost) and comfort with seeing past conversations (the privacy cost
# raised in the Gap section). "effort" is negatively worded, so it is
# reverse-scored in the analysis (8 - score) to make higher always better.
LIKERT = {1: "Strongly disagree", 2: "Disagree", 3: "Somewhat disagree", 4: "Neutral",
          5: "Somewhat agree", 6: "Agree", 7: "Strongly agree"}

QUESTIONNAIRE = [
    ("trust", "I trusted the assistant's personalised replies.", False),
    ("understanding", "I understood how the assistant used my memories.", False),
    ("predictability", "I could tell in advance when the assistant would use one of my memories.", False),
    ("control", "I felt in control of how the assistant used my memories.", False),
    ("effort", "Checking why a reply was personalised took a lot of effort.", True),
    ("comfort", "I am comfortable with an assistant showing me the past conversations "
                "my memories came from.", False),
]


def clean_comment(text: str, limit: int = 500) -> str:
    """Free-text comment made safe for the one-line, semicolon-separated log detail."""
    return " ".join((text or "").replace(";", ",").split())[:limit]
