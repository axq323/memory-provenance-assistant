"""Stored memories about the user, plus the conversations they came from.

The store meets the three requirements from Lecture 10: memories are
inspectable (plain-language text with a linked source), editable (one
action, and the edit persists) and forgettable (deletion removes the
entry; callers also purge derived copies such as context records).

A working copy lives in runtime/ (gitignored) so the seed data in data/
is never modified and the demo can always be reset.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "data" / "memories.seed.json"
SOURCES_PATH = ROOT / "data" / "sources.json"
RUNTIME_PATH = ROOT / "runtime" / "memories.json"


class MemoryStore:
    def __init__(self, path: Path = RUNTIME_PATH, seed_path: Path = SEED_PATH,
                 sources_path: Path = SOURCES_PATH):
        self.path = Path(path)
        self.seed_path = Path(seed_path)
        self.sources = json.loads(Path(sources_path).read_text(encoding="utf-8"))
        if not self.path.exists():
            self.reset()
        self._memories = json.loads(self.path.read_text(encoding="utf-8"))

    # ---- reading -------------------------------------------------------
    def all(self) -> list[dict]:
        return [dict(m) for m in self._memories]

    def get(self, memory_id: str) -> dict | None:
        for m in self._memories:
            if m["id"] == memory_id:
                return dict(m)
        return None

    def source_for(self, memory_id: str) -> dict | None:
        """The originating conversation, with the creating turn(s) flagged."""
        m = self.get(memory_id)
        if m is None or m.get("source_id") not in self.sources:
            return None
        src = self.sources[m["source_id"]]
        turns = [dict(t, creates_this_memory=memory_id in t.get("memory_ids", []))
                 for t in src["turns"]]
        return {"id": m["source_id"], "title": src["title"], "date": src["date"],
                "turns": turns}

    # ---- changing ------------------------------------------------------
    def edit(self, memory_id: str, new_text: str) -> None:
        new_text = new_text.strip()
        if not new_text:
            raise ValueError("Memory text cannot be empty.")
        m = self._find(memory_id)
        m["text"] = new_text
        m["edited"] = date.today().isoformat()
        self.save()

    def exclude_for_task(self, memory_id: str, task: str) -> None:
        """Scoped correction: keep the memory, but stop applying it to a task type."""
        m = self._find(memory_id)
        if task not in m["excluded_tasks"]:
            m["excluded_tasks"].append(task)
        self.save()

    def include_for_task(self, memory_id: str, task: str) -> None:
        """Undo for exclude_for_task."""
        m = self._find(memory_id)
        if task in m["excluded_tasks"]:
            m["excluded_tasks"].remove(task)
        self.save()

    def forget(self, memory_id: str) -> dict:
        m = self._find(memory_id)
        self._memories = [x for x in self._memories if x["id"] != memory_id]
        self.save()
        return m

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.seed_path, self.path)
        self._memories = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._memories, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ---- internal ------------------------------------------------------
    def _find(self, memory_id: str) -> dict:
        for m in self._memories:
            if m["id"] == memory_id:
                return m
        raise KeyError(f"No memory with id {memory_id!r}")
