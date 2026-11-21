"""Counterfactual memory check: measure which memories actually changed a reply.

Why: model self-report is not reliable. In testing, llama3.2:3b followed
the vegetarian memory in 12/12 runs but reported it in 0/12, so an
interface that trusts the model's memories_used list can tell users no
memory was used when one was. This check measures influence from
behaviour instead: remove each sent memory in turn, regenerate the
reply, and compare.

Live mode: every reply (baseline included) is regenerated at
temperature 0, so sampling noise is not mistaken for influence, and a
judge call decides whether the difference reflects the removed memory.
Replay mode: scripted replies are deterministic, so a reply that changes
when a memory is removed was influenced by it.

Cost: 1 + 2 x (memories sent) model calls in live mode, so the check
runs only on demand.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .context import assemble
from .llm import ModelError, ReplayModel
from .validate import validate

MAX_REPLY_CHARS = 1500  # keep judge prompts short

JUDGE_PROMPT = """You compare two replies to the same request.
Reply A was written knowing a fact about the user. Reply B was written without it.

FACT: {memory}
REQUEST: {request}

REPLY A (with the fact):
{a}

REPLY B (without the fact):
{b}

Does reply A reflect the fact in a way that reply B does not? Ignore differences
in wording that have nothing to do with the fact.
Reply ONLY with JSON: {{"influenced": true or false, "evidence": "<one short sentence>"}}"""


@dataclass
class Influence:
    id: str
    text: str
    reported: bool             # the model listed it in memories_used
    influenced: bool | None    # measured; None = could not decide
    evidence: str

    @property
    def verdict(self) -> str:
        if self.influenced is None:
            return "unclear"
        if self.influenced and not self.reported:
            return "unreported influence"
        if self.reported and not self.influenced:
            return "claimed, no measurable influence"
        return "agrees"

    def to_dict(self) -> dict:
        return {**asdict(self), "verdict": self.verdict}


def check_influence(model, user_text: str, memories: list[dict], task: str,
                    trigger: str | None, reported_ids: set[str],
                    history: list[dict] | None = None) -> list[Influence]:
    """Test every memory that context assembly would send for this request."""
    live = not isinstance(model, ReplayModel)
    full = assemble(user_text, memories, task, trigger, history)
    baseline = _answer(model, full, user_text, live)   # ModelError here propagates

    results = []
    for sent in full.memories_sent:
        mid = sent["id"]
        try:
            without = [m for m in memories if m["id"] != mid]
            record = assemble(user_text, without, task, trigger, history)
            alt = _answer(model, record, user_text, live)
            if live:
                influenced, evidence = _judge(model, sent["text"], user_text, baseline, alt)
            else:
                influenced = _normalise(alt) != _normalise(baseline)
                evidence = ("The reply changed when this memory was removed." if influenced
                            else "The reply was the same without this memory.")
        except ModelError as exc:
            influenced, evidence = None, f"Model call failed: {exc}"
        results.append(Influence(mid, sent["text"], mid in reported_ids, influenced, evidence))
    return results


def summarise(results: list[Influence]) -> dict:
    counts = {"agrees": 0, "unreported influence": 0,
              "claimed, no measurable influence": 0, "unclear": 0}
    for r in results:
        counts[r.verdict] += 1
    return counts


# ---------------------------------------------------------------- helpers ---
def _answer(model, record, user_text: str, live: bool) -> str:
    raw = model.generate(record, user_text, temperature=0.0) if live else model.generate(record, user_text)
    return validate(raw, record.sent_ids).answer


def _judge(model, memory: str, request: str, a: str, b: str) -> tuple[bool | None, str]:
    prompt = JUDGE_PROMPT.format(memory=memory, request=request,
                                 a=a[:MAX_REPLY_CHARS], b=b[:MAX_REPLY_CHARS])
    raw = model.complete([{"role": "user", "content": prompt}], temperature=0.0, json_mode=True)
    data = _parse_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("influenced"), bool):
        return None, "The judge's reply couldn't be read."
    return data["influenced"], str(data.get("evidence", "")).strip() or "No evidence given."


def _parse_json(raw: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _normalise(text: str) -> str:
    return " ".join(text.split()).lower()
