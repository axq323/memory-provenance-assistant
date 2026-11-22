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
so asking it cannot prime the behaviour that measures reliance.

Ground truth comes from the annotated replay scenarios (data/replay.json):
which memories fit each demo request and which would be misapplied. It is
looked up by prompt, so it also applies when the same prompts are asked in
live mode.
"""

from __future__ import annotations

from .llm import ReplayModel

_scenarios = ReplayModel()


def ground_truth(user_text: str) -> dict | None:
    """{"scenario", "fits", "misapplied"} for a demo prompt, or None if unknown."""
    s = _scenarios.match(user_text)
    if s is None:
        return None
    return {"scenario": s["id"], "fits": list(s.get("fits", [])),
            "misapplied": list(s.get("misapplied", []))}


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

    rows: interaction-log rows for this session, in order, from the task's
    prediction onwards. Returns counts plus the memory ids they corrected.
    """
    from datetime import datetime
    events = [r["event"] for r in rows]
    corrected = sorted({r["memory_id"] for r in rows if r["event"] in CORRECTION_EVENTS and r["memory_id"]})
    seconds = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
    return {"accepted": "reply_accepted" in events, "flagged": "reply_flagged" in events,
            "corrected": corrected, "sources_opened": events.count("source_opened"),
            "seconds": int(seconds)}


def reliance(misapplied_shown: set[str], fitting_shown: set[str], behaviour: dict) -> str:
    """Was the user's reaction appropriate (Assignment 1: accept suitable
    personalisation, question misapplied memories)?

    misapplied memory in the reply -> "appropriate" if they flagged the reply or
        corrected that memory, otherwise "over-reliance".
    only fitting memories          -> "under-reliance" if they flagged the reply or
        corrected a fitting memory, otherwise "appropriate".
    no annotated memory            -> "n/a".
    """
    corrected = set(behaviour["corrected"])
    if misapplied_shown:
        questioned = behaviour["flagged"] or bool(corrected & misapplied_shown)
        return "appropriate" if questioned else "over-reliance"
    if fitting_shown:
        questioned = behaviour["flagged"] or bool(corrected & fitting_shown)
        return "under-reliance" if questioned else "appropriate"
    return "n/a"


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
