import json

from mpa.validate import validate


def _raw(answer="Hi", used=()):
    return json.dumps({"answer": answer, "memories_used": list(used)})


def test_valid_claims_are_kept():
    r = validate(_raw(used=[{"id": "m1", "how": "kept it short"}]), {"m1", "m2"})
    assert r.parse_ok and r.used == [{"id": "m1", "how": "kept it short"}] and r.dropped == []


def test_claim_for_unsent_memory_is_dropped():
    r = validate(_raw(used=[{"id": "m1", "how": "x"}, {"id": "m9", "how": "y"}]), {"m1"})
    assert [u["id"] for u in r.used] == ["m1"]
    assert r.dropped == [{"id": "m9", "reason": "claimed by the model but not sent in context"}]


def test_duplicates_are_collapsed():
    r = validate(_raw(used=[{"id": "m1"}, {"id": "m1"}]), {"m1"})
    assert len(r.used) == 1


def test_code_fenced_json_and_bare_id_list_are_tolerated():
    raw = "```json\n" + _raw(used=["m2"]) + "\n```"
    r = validate(raw, {"m2"})
    assert r.parse_ok and r.used == [{"id": "m2", "how": ""}]


def test_json_with_surrounding_prose_is_extracted():
    r = validate("Sure! " + _raw(answer="Done") + " Hope that helps.", set())
    assert r.parse_ok and r.answer == "Done"


def test_unparseable_reply_is_shown_but_flagged():
    r = validate("Recursion is when a function calls itself.", {"m1"})
    assert not r.parse_ok
    assert r.answer.startswith("Recursion")
    assert r.used == [] and "couldn't be verified" in r.note


def test_malformed_memory_list_is_ignored_with_note():
    r = validate(json.dumps({"answer": "ok", "memories_used": "m1"}), {"m1"})
    assert r.parse_ok and r.used == [] and r.note
