"""Rule-based task-type classifier.

Scoped corrections ("don't use this memory for personal messages") need
a task type that is decided *before* the model call, so the excluded
memory never enters the context. A transparent keyword rule is used
rather than a model call: it is deterministic, testable, and the UI can
show which word triggered it ("birthday" -> personal messages). This
keeps the design at the simple end of the autonomy scale (Lecture 02).
"""

from __future__ import annotations

import re

# Order matters: the first matching rule wins.
RULES: list[tuple[str, list[str]]] = [
    ("personal_message", ["birthday", "anniversary", "congratulations message", "thank-you text",
                          "thank you text", "message for my friend", "message for my best friend",
                          "best friend", "to my grandmother", "to my mum", "to my mom", "to my dad"]),
    ("work_writing", ["my manager", "my boss", "work email", "professional email", "cover letter",
                      "email to", "colleague", "client"]),
    ("feedback", ["feedback", "my essay", "my draft", "proofread", "what should i change"]),
    ("cooking", ["dinner", "lunch", "breakfast", "recipe", "cook", "meal", "menu"]),
    ("explain", ["explain", "how does", "how do", "what is", "what are", "why does",
                 "understand", "teach me", "step by step"]),
    ("summarise", ["summarise", "summarize", "summary", "key points", "key ideas", "recap", "tl;dr"]),
    ("plan", ["plan", "schedule", "itinerary", "trip", "this week"]),
]

LABELS = {
    "personal_message": "personal messages",
    "work_writing": "work emails",
    "feedback": "feedback on your writing",
    "cooking": "meal ideas",
    "explain": "explanations",
    "summarise": "summaries",
    "plan": "planning",
    "other": "other requests",
}


def classify(text: str) -> tuple[str, str | None]:
    """Return (task_type, trigger_phrase). trigger_phrase is None for 'other'."""
    lowered = text.lower()
    for task, phrases in RULES:
        for phrase in phrases:
            if re.search(r"\b" + re.escape(phrase) + r"\b", lowered):
                return task, phrase
    return "other", None


def label(task: str) -> str:
    return LABELS.get(task, task)
