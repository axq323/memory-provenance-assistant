import json

from mpa.counterfactual import Influence, check_influence, summarise
from mpa.llm import ModelError, ReplayModel
from mpa.task_types import classify

DINNER = "Can you suggest something I could make for dinner tonight?"
BIRTHDAY = "Can you write a birthday message for my best friend Sam? Something warm and a bit funny."


def _check(model, store, text, reported):
    task, trigger = classify(text)
    return {r.id: r for r in check_influence(model, text, store.all(), task, trigger, set(reported))}


# ---------------------------------------------------------------- replay ---
def test_replay_detects_the_memories_that_shaped_the_dinner(store):
    res = _check(ReplayModel(), store, DINNER, reported={"m3", "m4"})
    assert res["m3"].influenced and res["m4"].influenced
    assert not res["m1"].influenced and not res["m2"].influenced
    assert all(r.verdict == "agrees" for r in res.values())


def test_unreported_influence_is_flagged(store):
    # simulate the llama3.2:3b failure: memory used but not reported
    res = _check(ReplayModel(), store, BIRTHDAY, reported=set())
    assert res["m2"].verdict == "unreported influence"


def test_claim_without_influence_is_flagged(store):
    res = _check(ReplayModel(), store, BIRTHDAY, reported={"m1"})
    assert res["m1"].verdict == "claimed, no measurable influence"


def test_excluded_memory_is_not_tested(store):
    store.exclude_for_task("m2", "personal_message")
    res = _check(ReplayModel(), store, BIRTHDAY, reported=set())
    assert "m2" not in res and set(res) == {"m1", "m3", "m4"}


def test_summary_counts_verdicts():
    rs = [Influence("m1", "", True, True, ""), Influence("m2", "", False, True, ""),
          Influence("m3", "", True, False, ""), Influence("m4", "", False, None, "")]
    assert summarise(rs) == {"agrees": 1, "unreported influence": 1,
                             "claimed, no measurable influence": 1, "unclear": 1}


# ------------------------------------------------------------ fake live ---
class FakeLive:
    """Vegetarian answer only when m3 is sent; judge says 'influenced' when replies differ."""

    name = "fake live"

    def __init__(self, judge_reply=None, fail_without=None):
        self.temperatures, self.judge_calls = [], 0
        self.judge_reply, self.fail_without = judge_reply, fail_without

    def generate(self, record, user_text, temperature=None):
        self.temperatures.append(temperature)
        if self.fail_without and self.fail_without not in record.sent_ids:
            raise ModelError("connection refused")
        meal = "chickpea curry" if "m3" in record.sent_ids else "chicken stir-fry"
        return json.dumps({"memories_used": [], "answer": f"Try a {meal}."})

    def complete(self, messages, temperature=None, json_mode=True):
        self.judge_calls += 1
        if self.judge_reply is not None:
            return self.judge_reply
        prompt = messages[0]["content"]
        a = prompt.split("REPLY A (with the fact):")[1].split("REPLY B")[0]
        b = prompt.split("REPLY B (without the fact):")[1].split("Does reply A")[0]
        changed = a.strip() != b.strip()
        return json.dumps({"influenced": changed, "evidence": "meat appears" if changed else "same"})


def test_live_uses_temperature_zero_and_one_judge_call_per_memory(store):
    fake = FakeLive()
    res = _check(fake, store, DINNER, reported=set())
    assert set(fake.temperatures) == {0.0}
    assert len(fake.temperatures) == 1 + 4 and fake.judge_calls == 4
    assert res["m3"].verdict == "unreported influence" and res["m3"].evidence == "meat appears"
    assert res["m1"].influenced is False


def test_unreadable_judge_reply_is_unclear(store):
    res = _check(FakeLive(judge_reply="not json"), store, DINNER, reported={"m3"})
    assert all(r.verdict == "unclear" for r in res.values())


def test_failed_regeneration_marks_only_that_memory_unclear(store):
    res = _check(FakeLive(fail_without="m2"), store, DINNER, reported=set())
    assert res["m2"].influenced is None and "failed" in res["m2"].evidence
    assert res["m3"].influenced is True


def test_live_model_complete_passes_temperature_and_json_mode():
    from mpa.llm import LiveModel

    class Completions:
        def create(self, **kwargs):
            self.kwargs = kwargs

            class R:
                pass
            r, c, m = R(), R(), R()
            m.content = "{}"
            c.message = m
            r.choices = [c]
            return r

    model = LiveModel("http://localhost:11434/v1", "x", "x")
    fake = Completions()
    model.client.chat.completions = fake
    model.complete([{"role": "user", "content": "hi"}], temperature=0.0, json_mode=False)
    assert fake.kwargs["temperature"] == 0.0 and "response_format" not in fake.kwargs
