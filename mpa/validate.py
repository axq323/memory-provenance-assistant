"""Check the model's reply before anything is shown to the user.

The model *claims* which memories it used. A schema only guarantees the
shape of that claim, not its truth, and a tool-style output
is a proposal the system must check. So a memory chip is
shown only if its ID was actually sent in the ContextRecord; any other
ID is dropped and reported in the developer view.

"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field


@dataclass
class ValidatedReply:
    answer: str
    used: list[dict] = field(default_factory=list)      # {id, how}, verified as sent
    dropped: list[dict] = field(default_factory=list)   # {id, reason}
    parse_ok: bool = True
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json(raw: str) -> dict | None:
    text = _FENCE.sub("", raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def validate(raw: str, sent_ids: set[str]) -> ValidatedReply:
    data = _extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return ValidatedReply(
            answer=raw.strip() or "(empty reply)", parse_ok=False,
            note="The reply wasn't in the expected format, so which memories it used couldn't be verified.")

    reply = ValidatedReply(answer=data["answer"].strip())
    claims = data.get("memories_used") or []
    if not isinstance(claims, list):
        claims = []
        reply.note = "The model's list of memories used was malformed and was ignored."

    seen = set()
    for c in claims:
        if isinstance(c, str):          # tolerate ["m1", "m2"]
            c = {"id": c, "how": ""}
        if not isinstance(c, dict) or "id" not in c:
            continue
        mid = str(c["id"]).strip()
        if mid in seen:
            continue
        seen.add(mid)
        if mid in sent_ids:
            reply.used.append({"id": mid, "how": str(c.get("how", "")).strip()})
        else:
            reply.dropped.append({"id": mid, "reason": "claimed by the model but not sent in context"})
    return reply
