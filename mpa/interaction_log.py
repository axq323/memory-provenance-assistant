"""Append-only CSV log of what the user did, for the planned study.

Behavioural reliance (accepting a reply, opening a source, correcting a
memory) is logged separately from anything self-reported, because the
two can diverge (Bucinca et al., 2021). The log stores memory IDs only,
never memory text, so forgetting a memory leaves no copy of it here.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "runtime" / "interaction_log.csv"

FIELDS = ["timestamp", "session_id", "condition", "mode", "turn_id", "task",
          "event", "memory_id", "detail"]

EVENTS = {
    "reply_shown", "source_opened", "source_closed", "not_for_task", "undo_not_for_task",
    "memory_edited", "memory_forgotten", "reply_accepted", "reply_flagged", "demo_reset",
    "influence_checked", "chat_cleared", "prediction_made",
}


def log_event(event: str, *, session_id: str, condition: str, mode: str,
              turn_id: str = "", task: str = "", memory_id: str = "", detail: str = "",
              path: Path = LOG_PATH) -> None:
    if event not in EVENTS:
        raise ValueError(f"Unknown event {event!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id, "condition": condition, "mode": mode,
            "turn_id": turn_id, "task": task, "event": event,
            "memory_id": memory_id, "detail": detail})


def read_log(path: Path = LOG_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
