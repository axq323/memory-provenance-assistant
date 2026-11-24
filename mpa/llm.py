"""Model back-ends. Both return raw text that goes through the validator.

ReplayModel (SIMULATED): scripted replies for the demo scenarios. The
reply is chosen from what context assembly actually sent, so the
user's corrections change the reply just as they would with a live
model. Scripting also means every demo and evaluation session sees the
same memory misapplication, which a live model would not guarantee.

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


def default_mode() -> str:
    """The mode a new session starts in: MODE in .env ("live" or "replay", default replay).
    The app can switch a session to the other mode (app.py, mode_changed)."""
    return "live" if os.getenv("MODE", "replay").strip().lower() == "live" else "replay"


def live_model_name() -> str:
    return os.getenv("LLM_MODEL", "qwen2.5:7b")


def get_model(mode: str | None = None):
    """Build the model for a mode ("replay" or "live"; default: MODE in .env, see .env.example)."""
    if (mode or default_mode()) == "live":
        return LiveModel(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            model=live_model_name(),
            api_key=os.getenv("LLM_API_KEY", ""))
    return ReplayModel()


def get_judge(mode: str | None = None):
    """The model that judges the counterfactual check (live mode only).

    JUDGE_MODEL can differ from LLM_MODEL, so the assistant isn't grading
    itself (e.g. llama3.2:3b as assistant, qwen2.5:7b as judge). Replay mode
    needs no judge, so this returns None there.
    """
    if (mode or default_mode()) != "live":
        return None
    return LiveModel(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        model=os.getenv("JUDGE_MODEL", "qwen2.5:7b"),
        api_key=os.getenv("LLM_API_KEY", ""),
        temperature=0.0)


def live_available(timeout: float = 3.0) -> tuple[bool, str]:
    """Can live mode be used right now? Asks the endpoint for its list of models, which costs
    no tokens. Returns (ok, message): the message says why not, or warns that the judge model
    is missing (replies still work then; only the influence check would fail)."""
    from openai import OpenAI
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model, judge = live_model_name(), os.getenv("JUDGE_MODEL", "qwen2.5:7b")
    try:
        client = OpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "") or "none",
                        timeout=timeout, max_retries=0)
        ids = {m.id for m in client.models.list()}
    except Exception as exc:  # not running, wrong URL, bad key...
        return False, (f"Couldn't reach the live model at {base_url} ({type(exc).__name__}). "
                       "Is Ollama running? Staying in replay mode.")
    if ids and model not in ids:
        return False, (f"The endpoint answered, but {model} isn't available there "
                       f"(with Ollama: ollama pull {model}). Staying in replay mode.")
    if ids and judge not in ids:
        return True, (f"Live mode on. The judge model {judge} isn't available, so the "
                      f"influence check won't run until you pull it.")
    return True, f"Live mode on: replies now come from {model}."