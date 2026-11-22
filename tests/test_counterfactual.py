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
    """Vegetarian answer only when m3 is sent. As a judge it answers 'follows'
    for the vegetarian fact only when the reply is meat-free (or as told)."""

    name = "fake live"
    model = "fake"

    def __init__(self, judge_reply=None, fail_without=None, follows=None):
        self.temperatures, self.judge_prompts = [], []
        self.judge_reply, self.fail_without = judge_reply, fail_without
        self.follows = follows          # optional fn(fact, reply) -> bool

    @property
    def judge_calls(self):
        return len(self.judge_prompts)

    def generate(self, record, user_text, temperature=None):
        self.temperatures.append(temperature)
        if self.fail_without and self.fail_without not in record.sent_ids:
            raise ModelError("connection refused")
        meal = "chickpea curry" if "m3" in record.sent_ids else "chicken stir-fry"
        return json.dumps({"memories_used": [], "answer": f"Try a {meal}."})

    def complete(self, messages, temperature=None, json_mode=True):
        prompt = messages[0]["content"]
        self.judge_prompts.append(prompt)
        if self.judge_reply is not None:
            return self.judge_reply
        fact = prompt.split("FACT ABOUT THE USER:")[1].split("\n")[0].strip()
        reply = prompt.split("REPLY:\n")[1].split("\n\nDoes this reply")[0]
        if self.follows:
            ok = self.follows(fact, reply)
        else:
            ok = "vegetarian" in fact and "chicken" not in reply
        return json.dumps({"follows": ok, "reason": "meat-free" if ok else "not related"})


def test_live_regenerates_once_per_memory_and_judges_each_reply(store):
    fake = FakeLive()
    res = _check(fake, store, DINNER, reported=set())
    assert set(fake.temperatures) == {0.0}
    assert len(fake.temperatures) == 1 + 4          # baseline regenerated (no shown reply given)
    assert fake.judge_calls == 8                     # 2 yes/no questions per memory
    assert res["m3"].verdict == "unreported influence"
    assert res["m3"].follows_with is True and res["m3"].follows_without is False
    assert res["m1"].influenced is False


def test_shown_reply_is_judged_instead_of_a_regenerated_one(store):
    fake = FakeLive()
    task, trigger = classify(DINNER)
    res = {r.id: r for r in check_influence(fake, DINNER, store.all(), task, trigger, set(),
                                            shown_answer="Try a lentil dal.")}
    assert len(fake.temperatures) == 4               # only the 'without' replies are generated
    assert any("Try a lentil dal." in p for p in fake.judge_prompts)
    assert res["m3"].influenced is True


def test_judge_sees_one_reply_and_is_not_told_which_version(store):
    fake = FakeLive()
    _check(fake, store, DINNER, reported=set())
    for p in fake.judge_prompts:
        assert p.count("REPLY:") == 1
        assert "with the fact" not in p.lower() and "without" not in p.lower()


def test_unrelated_difference_is_not_counted_as_influence(store):
    # the replies differ (chickpea vs chicken), but neither follows a formal-tone
    # fact, so the formal-tone memory must not be marked as influential
    res = _check(FakeLive(), store, DINNER, reported=set())
    assert res["m2"].influenced is False and res["m2"].verdict == "agrees"


def test_memory_the_reply_follows_anyway_made_no_difference(store):
    fake = FakeLive(follows=lambda fact, reply: "vegetarian" in fact)   # both replies 'follow'
    res = _check(fake, store, DINNER, reported={"m3"})
    assert res["m3"].influenced is False
    assert res["m3"].verdict == "claimed, no measurable influence"


def test_inconsistent_answers_are_unclear(store):
    # the reply WITH the memory doesn't follow it, the reply WITHOUT it does
    fake = FakeLive(follows=lambda fact, reply: "vegetarian" in fact and "chicken" in reply)
    res = _check(fake, store, DINNER, reported=set())
    assert res["m3"].verdict == "unclear" and "Inconsistent" in res["m3"].evidence


def test_unreadable_judge_reply_is_unclear(store):
    res = _check(FakeLive(judge_reply="not json"), store, DINNER, reported={"m3"})
    assert all(r.verdict == "unclear" for r in res.values())


def test_failed_regeneration_marks_only_that_memory_unclear(store):
    res = _check(FakeLive(fail_without="m2"), store, DINNER, reported=set())
    assert res["m2"].influenced is None and "failed" in res["m2"].evidence
    assert res["m3"].influenced is True


def test_separate_judge_model_answers_the_questions(store):
    assistant, judge = FakeLive(), FakeLive()
    task, trigger = classify(DINNER)
    check_influence(assistant, DINNER, store.all(), task, trigger, set(), judge=judge)
    assert assistant.judge_calls == 0 and judge.judge_calls == 8


def test_evidence_quotes_both_answers(store):
    res = _check(FakeLive(), store, DINNER, reported=set())
    assert res["m3"].evidence == "With it: meat-free. Without it: not related."


def test_calls_needed():
    from mpa.counterfactual import calls_needed
    assert calls_needed(4, live=True) == 12 and calls_needed(4, live=False) == 0


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
