"""Study instruments: ground truth and scoring for the outcomes in Assignment 1, Section 04.

  predictive understanding  -> score_prediction(): predicted vs actually used memories
  applicability judgement   -> fit questions, scored against ground_truth()   (Part 2)
  self-reported trust       -> end-of-session questionnaire, kept apart       (Part 3)

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
