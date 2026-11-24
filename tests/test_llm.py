import json

import pytest

from mpa.context import assemble
from mpa.llm import LiveModel, ModelError, ReplayModel
from mpa.task_types import classify
from mpa.validate import validate

EXPLAIN = "I don't really understand how solar panels work. Can you explain it properly, step by step?"


def _run(model, store, text):
    task, trig = classify(text)
    rec = assemble(text, store.all(), task, trig)
    return validate(model.generate(rec, text), rec.sent_ids)


def test_every_scenario_has_a_variant_for_every_subset():
    from itertools import combinations
    for s in ReplayModel().scenarios:
        subsets = {frozenset(c) for r in range(len(s["candidates"]) + 1)
                   for c in combinations(s["candidates"], r)}
        covered = {frozenset(u["id"] for u in v["uses"]) for v in s["variants"]}
        assert subsets == covered, s["id"]


def test_misapplication_scenario_uses_m1_by_default(store):
    reply = _run(ReplayModel(), store, EXPLAIN)
    assert {u["id"] for u in reply.used} == {"m1"}
    assert len(reply.answer) < 300  # the terse, over-applied reply


def test_not_for_this_task_changes_the_reply(store):
    store.exclude_for_task("m1", "explain")
    reply = _run(ReplayModel(), store, EXPLAIN)
    assert reply.used == []
    assert "step by step" in reply.answer and "inverter" in reply.answer


def test_formal_tone_misapplied_then_corrected(store):
    text = "Can you write a birthday message for my best friend Sam? Something warm and a bit funny."
    assert {u["id"] for u in _run(ReplayModel(), store, text).used} == {"m2"}
    store.exclude_for_task("m2", "personal_message")
    reply = _run(ReplayModel(), store, text)
    assert reply.used == [] and "Happy birthday, Sam" in reply.answer
    # still applies to work emails
    from mpa.context import assemble
    rec = assemble("Write an email to my manager", store.all(), "work_writing", "my manager")
    assert "m2" in rec.sent_ids


def test_forget_changes_the_dinner_suggestion(store):
    store.forget("m3")
    reply = _run(ReplayModel(), store, "Can you suggest something I could make for dinner tonight?")
    assert {u["id"] for u in reply.used} == {"m4"}
    assert "salmon" in reply.answer


def test_small_typos_still_match_but_unrelated_prompts_do_not():
    m = ReplayModel()
    assert m.match("can you suggest something i could make for dinner tonight") is not None
    assert m.match("What is the capital of France?") is None


def test_unmatched_prompt_returns_guidance_and_no_memories(store):
    reply = _run(ReplayModel(), store, "Tell me a joke")
    assert reply.used == [] and "Replay mode" in reply.answer


class _FakeCompletions:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.kwargs = content, exc, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc

        class R:  # minimal stand-in for the SDK response object
            pass
        r = R(); c = R(); msg = R()
        msg.content = self.content
        c.message = msg
        r.choices = [c] # type: ignore
        return r


def _live_with(fake):
    model = LiveModel("http://localhost:11434/v1", "test-model", "x")
    model.client.chat.completions = fake # type: ignore
    return model


def test_live_model_sends_assembled_messages_and_requests_json(store):
    fake = _FakeCompletions(content=json.dumps({"answer": "ok", "memories_used": []}))
    rec = assemble("hi", store.all(), "other", None)
    out = _live_with(fake).generate(rec, "hi")
    assert json.loads(out)["answer"] == "ok"
    assert fake.kwargs["messages"] == rec.messages
    assert fake.kwargs["response_format"] == {"type": "json_object"}


def test_live_model_wraps_errors(store):
    rec = assemble("hi", store.all(), "other", None)
    with pytest.raises(ModelError):
        _live_with(_FakeCompletions(exc=ConnectionError("refused"))).generate(rec, "hi")


def test_judge_is_none_in_replay_mode(monkeypatch):
    from mpa.llm import get_judge
    monkeypatch.setenv("MODE", "replay")
    assert get_judge() is None


def test_judge_model_comes_from_its_own_setting(monkeypatch):
    from mpa.llm import get_judge
    monkeypatch.setenv("MODE", "live")
    monkeypatch.setenv("LLM_MODEL", "llama3.2:3b")
    monkeypatch.setenv("JUDGE_MODEL", "qwen2.5:7b")
    judge = get_judge()
    assert judge.model == "qwen2.5:7b" and judge.temperature == 0.0 # type: ignore
    monkeypatch.delenv("JUDGE_MODEL")
    assert get_judge().model == "qwen2.5:7b"          # type: ignore # default judge


def test_work_email_scenario_uses_the_formal_tone_and_survives_the_birthday_correction(store):
    work = "Write an email to my manager asking for Friday afternoon off."
    store.exclude_for_task("m2", "personal_message")
    reply = _run(ReplayModel(), store, work)
    assert {u["id"] for u in reply.used} == {"m2"} and "Dear Ms Patel" in reply.answer
    store.forget("m2")
    assert _run(ReplayModel(), store, work).used == []
