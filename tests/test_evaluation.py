from mpa.evaluation import ground_truth, score_prediction


def test_ground_truth_for_demo_prompts():
    assert ground_truth("Can you suggest something I could make for dinner tonight?") == \
        {"scenario": "S1-dinner", "fits": ["m3", "m4"], "misapplied": []}
    assert ground_truth("Can you write a birthday message for my best friend Sam? "
                        "Something warm and a bit funny.")["misapplied"] == ["m2"]
    assert ground_truth("Tell me a joke") is None


def test_prediction_scoring():
    assert score_prediction({"m3", "m4"}, {"m3", "m4"}) == {"exact": True, "jaccard": 1.0}
    assert score_prediction({"m3"}, {"m3", "m4"}) == {"exact": False, "jaccard": 0.5}
    assert score_prediction(set(), set()) == {"exact": True, "jaccard": 1.0}     # "none of them", correctly
    assert score_prediction({"m1"}, set()) == {"exact": False, "jaccard": 0.0}


# ------------------------------------------------------------ Part 2 ---
from mpa.evaluation import reliance, score_fit, summarise_task  # noqa: E402

BIRTHDAY_TRUTH = {"scenario": "S3-birthday", "fits": [], "misapplied": ["m2"]}
DINNER_TRUTH = {"scenario": "S1-dinner", "fits": ["m3", "m4"], "misapplied": []}


def test_fit_scoring_against_ground_truth():
    assert score_fit("no", "m2", BIRTHDAY_TRUTH) == {"truth": "misapplied", "correct": True, "unsure": False}
    assert score_fit("yes", "m2", BIRTHDAY_TRUTH)["correct"] is False
    assert score_fit("yes", "m3", DINNER_TRUTH)["correct"] is True
    assert score_fit("unsure", "m3", DINNER_TRUTH) == {"truth": "fits", "correct": False, "unsure": True}
    assert score_fit("yes", "m1", None) == {"truth": "unknown", "correct": None, "unsure": False}


def _row(event, memory_id=""):
    return {"event": event, "memory_id": memory_id}


def test_task_summary_counts_what_the_user_did():
    rows = [_row("source_opened", "m2"), _row("source_closed", "m2"), _row("not_for_task", "m2"),
            _row("reply_shown"), _row("source_opened", "m3"), _row("reply_accepted")]
    s = summarise_task(rows, "2026-11-22T10:00:00", "2026-11-22T10:01:05")
    assert s == {"accepted": True, "flagged": False, "corrected": ["m2"], "decision": "accepted+corrected",
                 "sources_opened": 2, "seconds": 65}
    assert summarise_task([_row("source_opened", "m2")], "2026-11-22T10:00:00",
                          "2026-11-22T10:00:10")["decision"] == "none"


def _b(accepted=False, flagged=False, corrected=()):
    return {"accepted": accepted, "flagged": flagged, "corrected": list(corrected)}


def test_reliance_outcomes():
    # misapplied memory in the reply: questioning it is appropriate
    assert reliance({"m2"}, set(), _b(flagged=True)) == "appropriate"
    assert reliance({"m2"}, set(), _b(corrected=["m2"])) == "appropriate"
    assert reliance({"m2"}, set(), _b(accepted=True)) == "over-reliance"
    assert reliance({"m2"}, set(), _b()) == "no decision"                       # did nothing
    assert reliance({"m2"}, set(), _b(corrected=["m3"])) == "over-reliance"     # corrected the wrong memory
    assert reliance(set(), {"m3", "m4"}, _b()) == "no decision"
    # only fitting memories: accepting is appropriate, questioning them is under-reliance
    assert reliance(set(), {"m3", "m4"}, _b(accepted=True)) == "appropriate"
    assert reliance(set(), {"m3", "m4"}, _b(corrected=["m4"])) == "under-reliance"
    assert reliance(set(), {"m3", "m4"}, _b(flagged=True)) == "under-reliance"
    assert reliance(set(), set(), _b()) == "n/a"


# ------------------------------------------------------------ Part 3 ---
from mpa.evaluation import LIKERT, QUESTIONNAIRE, clean_comment  # noqa: E402


def test_questionnaire_items_are_unique_and_only_effort_is_reversed():
    ids = [i for i, _, _ in QUESTIONNAIRE]
    assert len(ids) == len(set(ids)) == 6
    assert [i for i, _, rev in QUESTIONNAIRE if rev] == ["effort"]
    assert list(LIKERT) == [1, 2, 3, 4, 5, 6, 7]


def test_comments_are_made_safe_for_the_log():
    assert clean_comment("Good;\nbut slow  ") == "Good, but slow"
    assert len(clean_comment("x" * 900)) == 500


# ------------------------------------------------------------ corrections ---
from mpa.evaluation import correction_actions, correction_outcome, final_state  # noqa: E402

START = {"text": "You want a formal, professional tone.", "excluded_tasks": []}


def test_final_state_after_the_task():
    assert final_state(START, None, "personal_message") == "forgotten"
    assert final_state(START, dict(START), "personal_message") == "unchanged"
    assert final_state(START, {**START, "excluded_tasks": ["personal_message"]}, "personal_message") == "restricted"
    assert final_state(START, {**START, "excluded_tasks": ["plan"]}, "personal_message") == "restricted elsewhere"
    assert final_state(START, {"text": "Formal for work.", "excluded_tasks": []}, "personal_message") == "edited"
    # a restriction that was already there before the task is not credited to it
    before = {**START, "excluded_tasks": ["personal_message"]}
    assert final_state(before, dict(before), "personal_message") == "unchanged"


def test_correction_outcomes():
    m = {"m2"}
    assert correction_outcome(m, set(), {"m2": "restricted"}, ["restrict:m2@personal_message"]) == "scoped"
    assert correction_outcome(m, set(), {"m2": "forgotten"}, ["forget:m2"]) == "forgot"
    assert correction_outcome(m, set(), {"m2": "edited"}, ["edit:m2"]) == "edited"
    assert correction_outcome(m, set(), {"m2": "restricted elsewhere"}, ["restrict:m2@plan"]) == "wrong scope"
    assert correction_outcome(m, set(), {"m2": "unchanged"},
                              ["restrict:m2@personal_message", "undo:m2@personal_message"]) == "undone"
    assert correction_outcome(m, set(), {"m2": "unchanged"}, []) == "none"
    assert correction_outcome(set(), {"m3"}, {"m3": "forgotten"}, ["forget:m3"]) == "unneeded"
    assert correction_outcome(set(), {"m3"}, {"m3": "unchanged"}, []) == "none"
    assert correction_outcome(set(), set(), {}, []) == "n/a"


def test_correction_actions_keep_order_and_scope():
    rows = [{"event": "not_for_task", "memory_id": "m2", "task": "personal_message"},
            {"event": "source_opened", "memory_id": "m2", "task": ""},
            {"event": "undo_not_for_task", "memory_id": "m2", "task": "personal_message"},
            {"event": "memory_forgotten", "memory_id": "m2", "task": ""}]
    assert correction_actions(rows) == ["restrict:m2@personal_message", "undo:m2@personal_message", "forget:m2"]
