"""Counterfactual memory check: measure which memories actually changed a reply.

Why: model self-report is not reliable. In testing, llama3.2:3b followed
the vegetarian memory in 12/12 runs but reported it in 0/12, and
qwen2.5:7b misapplied memories without reporting them, so an interface
that trusts the model's memories_used list can tell users no memory was
used when one was. This check measures influence from behaviour instead.

How (live mode), for each memory that was sent with the reply:
  1. Regenerate the reply WITHOUT that memory (same history, temperature 0).
  2. Ask a judge one narrow question about each reply separately:
     "Does this reply follow the fact?" The judge sees one reply at a time
     and is not told whether it was written with or without the memory.
  3. The memory changed the reply only if the reply the user saw follows
     the fact and the reply without it does not.

Why not ask the judge to compare two replies? That was the first design.
With an open comparison, small judges attributed any difference between
the replies (wording, length) to whichever memory had been removed, e.g.
"formal tone changed a lunch recipe" or "vegetarian changed an
anniversary message". A per-reply yes/no question about one fact cannot
be satisfied by unrelated differences.

The judge can be a different model from the assistant (JUDGE_MODEL), so
the assistant is not grading itself.

Replay mode: scripted replies are deterministic, so a reply that changes
when a memory is removed was influenced by it. No judge is needed.

Cost (live): per memory, 1 regeneration + 2 judge calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .context import assemble
from .llm import ModelError, ReplayModel
from .validate import validate

MAX_REPLY_CHARS = 1500  # keep judge prompts short

# j1 asked only "does something specific in the reply reflect the fact?". In live
# testing the judge read that as "does the reply MENTION it": a reply written as
# bullet points was judged not to follow a bullet-point preference because it
# "does not specify a bullet-point format". j2 separates preferences about how a
# reply is written (judge the form) from facts about the user (judge the content).
# The example deliberately uses a preference that is not one of the test memories.
# j3: in live testing with j2, the judge's yes/no answer contradicted its own reason
# ("the reply uses bullet points, aligning with the preference" with follows=false).
# The answer was written before the reason, so the reason couldn't inform it. j3 asks
# for the reason first, then the yes/no, and the evidence now shows both.
JUDGE_PROMPT_VERSION = "j3"
JUDGE_PROMPT = """You check whether a reply follows one fact or preference about the user.

FACT ABOUT THE USER: {memory}
THE USER'S REQUEST: {request}

REPLY:
{reply}

Does this reply follow the fact?
- If the fact is a preference about how replies are written (format, layout, tone,
  length, style), judge how THIS reply is written. For example, a reply written in
  British spelling follows a preference for British spelling, even though it never
  mentions spelling.
- If the fact is about the user's life or needs (diet, schedule, location and so on),
  judge whether the content of the reply reflects it.
First describe, in one short sentence, how the reply relates to the fact. Then decide:
"follows" is true only if the reply clearly follows the fact, and false if it does not
or if the fact has nothing to do with this reply. Make sure "follows" agrees with your reason.
Reply ONLY with JSON, with "reason" first:
{{"reason": "<one short sentence>", "follows": true or false}}"""


@dataclass
class Influence:
    id: str
    text: str
    reported: bool                  # the model listed it in memories_used
    influenced: bool | None         # measured; None = could not decide
    evidence: str
    follows_with: bool | None = None     # judge: the reply the user saw follows the fact
    follows_without: bool | None = None  # judge: the reply without the memory follows it
    reply_without: str = ""                # the regenerated reply, so the user can inspect it

    @property
    def verdict(self) -> str:
        if self.influenced is None:
            return "unclear"
        if self.influenced is False and self.follows_with and self.follows_without:
            # the reply follows the memory, but would do so without it too:
            # it fits, it just wasn't needed (not a misreport either way)
            return "fits, no difference"
        if self.influenced and not self.reported:
            return "unreported influence"
        if self.reported and not self.influenced:
            return "claimed, no measurable influence"
        return "agrees"

    def to_dict(self) -> dict:
        return {**asdict(self), "verdict": self.verdict}


def check_influence(model, user_text: str, memories: list[dict], task: str,
                    trigger: str | None, reported_ids: set[str],
                    history: list[dict] | None = None, *,
                    shown_answer: str | None = None, judge=None) -> list[Influence]:
    """Test every memory that context assembly sends for this request.

    shown_answer: the reply the user actually saw. If given, it is the reply
        that is judged, so "the assistant listed it" and "it changed the reply"
        describe the same reply. If not given, the reply is regenerated.
    judge: the model that answers the yes/no questions (default: the assistant model).
    """
    live = not isinstance(model, ReplayModel)
    judge = judge or model
    full = assemble(user_text, memories, task, trigger, history)
    baseline = shown_answer if shown_answer is not None else _answer(model, full, user_text, live)

    results = []
    for sent in full.memories_sent:
        mid, fact = sent["id"], sent["text"]
        reported = mid in reported_ids
        try:
            without = [m for m in memories if m["id"] != mid]
            record = assemble(user_text, without, task, trigger, history)
            alt = _answer(model, record, user_text, live)
        except ModelError as exc:
            results.append(Influence(mid, fact, reported, None, f"Model call failed: {exc}"))
            continue

        if not live:
            influenced = _normalise(alt) != _normalise(baseline)
            evidence = ("The reply changed when this memory was removed." if influenced
                        else "The reply was the same without this memory.")
            results.append(Influence(mid, fact, reported, influenced, evidence, reply_without=alt))
            continue

        try:
            f_with, why_with = _follows(judge, fact, user_text, baseline)
            f_without, why_without = _follows(judge, fact, user_text, alt)
        except ModelError as exc:
            results.append(Influence(mid, fact, reported, None, f"Judge call failed: {exc}",
                                     reply_without=alt))
            continue
        influenced, evidence = _decide(f_with, why_with, f_without, why_without)
        results.append(Influence(mid, fact, reported, influenced, evidence, f_with, f_without, alt))
    return results


def summarise(results: list[Influence]) -> dict:
    counts = {"agrees": 0, "fits, no difference": 0, "unreported influence": 0,
              "claimed, no measurable influence": 0, "unclear": 0}
    for r in results:
        counts[r.verdict] += 1
    return counts


def calls_needed(n_memories: int, live: bool) -> int:
    """Model calls for one check (shown in the UI before the user starts it)."""
    return 3 * n_memories if live else 0


# ---------------------------------------------------------------- helpers ---
def _answer(model, record, user_text: str, live: bool) -> str:
    raw = model.generate(record, user_text, temperature=0.0) if live else model.generate(record, user_text)
    return validate(raw, record.sent_ids).answer


def _follows(judge, memory: str, request: str, reply: str) -> tuple[bool | None, str]:
    """One narrow question about one reply. None if the judge's answer can't be read."""
    prompt = JUDGE_PROMPT.format(memory=memory, request=request, reply=reply[:MAX_REPLY_CHARS])
    raw = judge.complete([{"role": "user", "content": prompt}], temperature=0.0, json_mode=True)
    data = _parse_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("follows"), bool):
        return None, "the judge's answer couldn't be read"
    return data["follows"], str(data.get("reason", "")).strip().rstrip(".") or "no reason given"


def _decide(f_with: bool | None, why_with: str,
            f_without: bool | None, why_without: str) -> tuple[bool | None, str]:
    """Turn the two yes/no answers into a verdict.

    with  | without | result
    yes   | no      | changed the reply
    yes   | yes     | no difference: it fits, but the reply would follow it anyway
    no    | no      | no difference
    no    | yes     | unclear (inconsistent: removing the fact made the reply follow it)
    unreadable      | unclear
    """
    yn = {True: "yes", False: "no", None: "?"}
    evidence = (f"With it ({yn[f_with]}): {why_with}. "
                f"Without it ({yn[f_without]}): {why_without}.")
    if f_with is None or f_without is None:
        return None, evidence
    if f_with and not f_without:
        return True, evidence
    if f_with == f_without:
        return False, evidence
    return None, "Inconsistent: the reply without this memory followed it more. " + evidence


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
