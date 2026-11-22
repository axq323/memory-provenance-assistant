"""Model back-ends. Both return raw text that goes through the validator.

ReplayModel (SIMULATED): scripted replies for the demo scenarios. The
reply is chosen from what context assembly actually sent, so the
user's corrections change the reply just as they would with a live
model. Scripting also guarantees every study participant sees the same
memory misapplication, which a live model would not.

LiveModel (IMPLEMENTED): any OpenAI-compatible chat endpoint, such as a
local Ollama server (free, no key) or a hosted API with the reviewer's
own key. Configuration comes from environment variables (.env).
"""

from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

from .context import ContextRecord

ROOT = Path(__file__).resolve().parent.parent
REPLAY_PATH = ROOT / "data" / "replay.json"


class ModelError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class ReplayModel:
    name = "replay (scripted demo replies)"

    def __init__(self, path: Path = REPLAY_PATH):
        self.scenarios = json.loads(Path(path).read_text(encoding="utf-8"))["scenarios"]

    def match(self, user_text: str) -> dict | None:
        target = _normalise(user_text)
        best, best_score = None, 0.0
        for s in self.scenarios:
            score = difflib.SequenceMatcher(None, target, _normalise(s["prompt"])).ratio()
            if score > best_score:
                best, best_score = s, score
        return best if best_score >= 0.85 else None

    def generate(self, record: ContextRecord, user_text: str, temperature: float | None = None) -> str:
        # temperature is accepted for a common interface; scripted replies ignore it
        scenario = self.match(user_text)
        if scenario is None:
            return json.dumps({
                "answer": ("Replay mode only covers the demo prompts shown above the chat. "
                           "Pick one of those, or switch to live mode (see the README) to ask anything."),
                "memories_used": []})
        present = set(scenario["candidates"]) & record.sent_ids
        variant = self._pick_variant(scenario, present)
        return json.dumps({"answer": variant["answer"], "memories_used": variant["uses"]})

    @staticmethod
    def _pick_variant(scenario: dict, present: set[str]) -> dict:
        # Exact match on the set of candidate memories present; otherwise the
        # variant using the most memories that are all present.
        for v in scenario["variants"]:
            if {u["id"] for u in v["uses"]} == present:
                return v
        usable = [v for v in scenario["variants"] if {u["id"] for u in v["uses"]} <= present]
        return max(usable, key=lambda v: len(v["uses"]))


class LiveModel:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.3):
        from openai import OpenAI  # imported lazily so replay mode needs no network setup
        self.client = OpenAI(base_url=base_url, api_key=api_key or "none", timeout=120)
        self.model = model
        self.temperature = temperature
        self.name = f"live ({model} at {base_url})"

    def generate(self, record: ContextRecord, user_text: str, temperature: float | None = None) -> str:
        return self.complete(record.messages,
                             temperature=self.temperature if temperature is None else temperature)

    def complete(self, messages: list[dict], temperature: float | None = None,
                 json_mode: bool = True) -> str:
        """One chat call with arbitrary messages (also used by the counterfactual judge)."""
        kwargs = {"model": self.model, "messages": messages,
                  "temperature": self.temperature if temperature is None else temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # network, auth, unknown model...
            raise ModelError(f"The model call failed: {type(exc).__name__}: {exc}") from exc
        return resp.choices[0].message.content or ""


def get_model():
    """Build the model from environment configuration (see .env.example)."""
    mode = os.getenv("MODE", "replay").strip().lower()
    if mode == "live":
        return LiveModel(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
            api_key=os.getenv("LLM_API_KEY", ""))
    return ReplayModel()


def get_judge():
    """The model that judges the counterfactual check (live mode only).

    JUDGE_MODEL can differ from LLM_MODEL, so the assistant isn't grading
    itself (e.g. llama3.2:3b as assistant, qwen2.5:7b as judge). Replay mode
    needs no judge, so this returns None there.
    """
    if os.getenv("MODE", "replay").strip().lower() != "live":
        return None
    return LiveModel(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        model=os.getenv("JUDGE_MODEL", "qwen2.5:7b"),
        api_key=os.getenv("LLM_API_KEY", ""),
        temperature=0.0)