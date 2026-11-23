"""Headless UI tests using Streamlit's AppTest (replay mode, temp runtime dir)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
EXPLAIN = "I don't really understand how solar panels work. Can you explain it properly, step by step?"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "replay")
    monkeypatch.setenv("MPA_RUNTIME_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _all_logs(tmp_path):
    """Every session's log, concatenated (each session writes its own file)."""
    return "".join(p.read_text() for p in sorted((tmp_path / "sessions").glob("*/interaction_log.csv")))


def _button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def test_replay_banner_and_demo_buttons(app):
    assert any("Replay mode" in i.value for i in app.info)
    assert len([b for b in app.button if (b.key or "").startswith("demo_")]) == 3


def test_demo_prompt_shows_reply_with_memory_chips(app):
    _button(app, "I don't really understand how solar").click().run()
    texts = " ".join(m.value for m in app.markdown)
    assert "Memories used in this reply" in texts
    assert "short bullet-point summaries" in texts


def test_show_source_reveals_originating_turn(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("src_") and "panel" not in b.key).click().run()
    assert not app.exception
    assert any("saved from this message" in i.value for i in app.info)


def test_not_for_this_task_regenerates_without_the_memory(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("nft_") and (b.key or "").endswith(":m1")).click().run()
    assert not app.exception
    assert any("won't use" in s.value for s in app.success)
    texts = " ".join(m.value for m in app.markdown)
    assert "**5. An inverter converts it for your home.**" in texts   # detailed reply now shown
    assert any("Not used for explanations" in c.value for c in app.caption)


def test_undo_restores_the_memory_for_that_task(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("nft_") and (b.key or "").endswith(":m1")).click().run()
    next(b for b in app.button if (b.key or "").startswith("undo_")).click().run()
    assert not app.exception
    assert any(c.value == "Undone." for c in app.caption)
    assert not any((b.key or "").startswith("again_") for b in app.button)


def test_forget_removes_memory_everywhere(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("forget_") and (b.key or "").endswith(":m1")).click().run()
    assert not app.exception
    assert any("since been forgotten" in c.value for c in app.caption)
    record = app.session_state["messages"][-1]["record"]
    assert "short bullet-point" not in str(record.to_dict())


def test_edit_updates_the_memory(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    key = next(t.key for t in app.text_area if t.key.endswith(":m1"))
    app.text_area(key=key).set_value("You prefer bullet points for essay feedback.")
    app.button(key=key.replace("edit_text_", "edit_save_")).click().run()
    assert not app.exception
    assert "bullet points for essay feedback" in " ".join(m.value for m in app.markdown)


def test_condition_b_hides_source_context(app):
    app.radio(key="condition").set_value("B").run()
    app.chat_input[0].set_value(EXPLAIN).run()
    assert not app.exception
    assert not any((b.key or "").startswith("src_") for b in app.button)
    assert any((b.key or "").startswith("nft_") for b in app.button)   # controls remain


def test_verdict_is_logged(app, tmp_path):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("flag_")).click().run()
    log = _all_logs(tmp_path)
    assert "reply_flagged" in log and "used=m1" in log


DINNER = "Can you suggest something I could make for dinner tonight?"


def test_influence_check_shows_measured_memories(app, tmp_path):
    app.chat_input[0].set_value(DINNER).run()
    next(b for b in app.button if (b.key or "").startswith("infl_")).click().run()
    assert not app.exception
    texts = " ".join(m.value for m in app.markdown)
    assert "Which memories mattered" in texts
    assert texts.count("Used, as the assistant said") == 2       # m3, m4 (replay reports honestly)
    assert texts.count("Not used, as the assistant said") == 2   # m1, m2
    captions = " ".join(c.value for c in app.caption)
    assert "removing it changed the reply" in captions and "removing it made no difference" in captions
    log = _all_logs(tmp_path)
    assert "influence_checked" in log and "m3=agrees" in log


def test_influence_check_respects_exclusions(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("nft_") and (b.key or "").endswith(":m1")).click().run()
    msg_index = len(app.session_state["messages"]) - 1
    next(b for b in app.button if (b.key or "").startswith("infl_")
         and (b.key or "").endswith(app.session_state["messages"][msg_index]["turn_id"])).click().run()
    assert not app.exception
    tested = {r.id for r in app.session_state["messages"][msg_index]["influence"]}
    assert "m1" not in tested                      # excluded for explanations, so never sent


def test_forget_redacts_influence_results(app):
    app.chat_input[0].set_value(DINNER).run()
    next(b for b in app.button if (b.key or "").startswith("infl_")).click().run()
    next(b for b in app.button if (b.key or "").startswith("forget_") and (b.key or "").endswith(":m3")).click().run()
    assert not app.exception
    results = app.session_state["messages"][-1]["influence"]
    assert all("vegetarian" not in r.text for r in results)


def test_influence_toggle_hides_the_check(app):
    app.toggle(key="influence_check").set_value(False).run()
    app.chat_input[0].set_value(DINNER).run()
    assert not any((b.key or "").startswith("infl_") for b in app.button)


# ------------------------------------------------- stage 1: history and check fixes ---
def _assistant(at):
    return [m for m in at.session_state["messages"] if m["role"] == "assistant"]


def test_each_reply_stores_the_history_it_was_sent(app):
    app.chat_input[0].set_value(DINNER).run()
    app.chat_input[0].set_value(EXPLAIN).run()
    first, second = _assistant(app)
    assert first["history"] == []
    assert [h["role"] for h in second["history"]] == ["user", "assistant"]
    assert second["history"][0]["content"] == DINNER


def test_regeneration_reuses_the_original_history_not_the_bad_reply(app):
    app.chat_input[0].set_value(DINNER).run()
    app.chat_input[0].set_value(EXPLAIN).run()
    original = _assistant(app)[1]
    next(b for b in app.button if (b.key or "") == f"nft_{original['turn_id']}:m1").click().run()
    redo = _assistant(app)[-1]
    assert redo["regenerated_from"] == original["turn_id"]
    assert redo["history"] == original["history"]
    assert original["reply"].answer not in [h["content"] for h in redo["history"]]


def test_corrected_reply_is_left_out_of_later_history(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    original = _assistant(app)[0]
    next(b for b in app.button if (b.key or "") == f"nft_{original['turn_id']}:m1").click().run()
    redo = _assistant(app)[-1]
    app.chat_input[0].set_value(DINNER).run()
    contents = [h["content"] for h in _assistant(app)[-1]["history"]]
    assert original["reply"].answer not in contents
    assert redo["reply"].answer in contents
    assert contents.count(EXPLAIN) == 1


def test_influence_check_tests_the_memories_as_sent(app):
    # m3 is excluded for meal ideas AFTER the first dinner reply; checking that
    # first reply must still test m3, because m3 was sent with it.
    app.chat_input[0].set_value(DINNER).run()
    original = _assistant(app)[0]
    next(b for b in app.button if (b.key or "") == f"nft_{original['turn_id']}:m3").click().run()
    app.button(key=f"infl_{original['turn_id']}").click().run()
    assert not app.exception
    assert {r.id for r in original["influence"]} == {"m1", "m2", "m3", "m4"}


def test_clear_chat_keeps_memories_and_corrections(app, tmp_path):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("nft_") and (b.key or "").endswith(":m1")).click().run()
    app.button(key="clear_chat").click().run()
    assert not app.exception
    assert app.session_state["messages"] == []
    assert any("Not used for explanations" in c.value for c in app.caption)   # correction kept
    assert "chat_cleared" in _all_logs(tmp_path)


def test_no_report_caption_does_not_claim_no_memories_were_used(app):
    app.chat_input[0].set_value("Tell me a joke").run()
    captions = [c.value for c in app.caption]
    assert "The assistant didn't report using any memories." in captions
    assert not any("No stored memories were used" in c for c in captions)


def _submit_prediction(app):
    # tick boxes and submit in the same run, as a real form submission does
    next(b for b in app.button if b.label == "Show me the reply").click().run()


def _evaluation(app):
    app.toggle(key="evaluation_mode").set_value(True).run()
    return app


def test_evaluation_mode_asks_for_a_prediction_before_the_reply(app):
    _evaluation(app)
    _button(app, "Can you write a birthday message").click().run()
    assert not app.exception
    assert any("which of your memories do you expect" in m.value for m in app.markdown)
    assert not any(m["role"] == "assistant" for m in app.session_state["messages"])   # no reply yet


def test_prediction_is_logged_against_what_was_used(app, tmp_path):
    _evaluation(app)
    _button(app, "Can you write a birthday message").click().run()
    app.checkbox(key="pred_m2").check()
    _submit_prediction(app)
    assert not app.exception
    reply = app.session_state["messages"][-1]
    assert reply["role"] == "assistant" and reply["prediction"]["exact"] is True
    log = _all_logs(tmp_path)
    assert "prediction_made" in log
    assert "predicted=m2;used=m2;exact=True;jaccard=1.0;scenario=S3-birthday" in log


def test_none_of_them_and_empty_prediction(app, tmp_path):
    _evaluation(app)
    _button(app, "Can you suggest something I could make").click().run()
    _submit_prediction(app)
    assert any("Tick at least one" in w.value for w in app.warning)               # nothing ticked
    app.checkbox(key="pred_none").check()
    _submit_prediction(app)
    assert not app.exception
    log = _all_logs(tmp_path)
    assert "predicted=none;used=m3,m4;exact=False;jaccard=0.0;scenario=S1-dinner" in log


def test_evaluation_mode_hides_the_influence_check(app):
    _evaluation(app)
    _button(app, "Can you suggest something I could make").click().run()
    app.checkbox(key="pred_m3").check()
    _submit_prediction(app)
    assert not any((b.key or "").startswith("infl_") for b in app.button)


def test_evaluation_mode_from_url(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "replay")
    monkeypatch.setenv("MPA_RUNTIME_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["evaluate"] = "1"
    at.run()
    assert at.session_state["evaluation_mode"] is True


def test_clear_chat_cancels_a_pending_prediction(app):
    app.toggle(key="evaluation_mode").set_value(True).run()
    app.chat_input[0].set_value(DINNER).run()
    assert app.session_state["awaiting_prediction"] == DINNER
    app.button(key="clear_chat").click().run()
    assert not app.exception
    assert app.session_state["awaiting_prediction"] is None
    assert app.session_state["messages"] == []


# ------------------------------------------------- stage 3: memory statuses ---
def test_sent_but_unclaimed_memories_are_listed_with_controls(app):
    app.chat_input[0].set_value(EXPLAIN).run()          # replay: assistant claims m1 only
    turn = _assistant(app)[0]["turn_id"]
    assert any("Also sent with this request, not listed by the assistant (3)" in e.label
               for e in app.expander)
    keys = {b.key for b in app.button}
    assert {f"nft_{turn}:m2", f"nft_{turn}:m3", f"nft_{turn}:m4"} <= keys
    assert any("(as reported by the assistant)" in m.value for m in app.markdown)


def test_unclaimed_memory_can_be_corrected_and_is_then_withheld(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    turn = _assistant(app)[0]["turn_id"]
    app.button(key=f"nft_{turn}:m2").click().run()
    assert not app.exception
    redo = _assistant(app)[-1]
    assert redo["regenerated_from"] == turn
    assert "m2" not in redo["record"].sent_ids
    assert any(c.value.startswith("⛔ Not sent: You want a formal")
               and "you chose not to use it for explanations" in c.value for c in app.caption)


def test_check_results_appear_on_the_memory_chips(app):
    app.chat_input[0].set_value(DINNER).run()
    turn = _assistant(app)[0]["turn_id"]
    app.button(key=f"infl_{turn}").click().run()
    captions = [c.value for c in app.caption]
    assert captions.count("Measured by the check: ✅ Used, as the assistant said") == 2      # m3, m4
    assert captions.count("Measured by the check: ✅ Not used, as the assistant said") == 2  # m1, m2


def test_unreported_influence_opens_the_unclaimed_list_with_a_warning(app):
    from mpa.counterfactual import Influence
    app.chat_input[0].set_value(DINNER).run()
    msg = _assistant(app)[0]
    msg["influence"] = [Influence("m2", "formal", False, True, "x")]
    app.run()
    exp = next(e for e in app.expander if "Also sent with this request" in e.label)
    assert exp.label.startswith("⚠️") and exp.proto.expanded
    assert any("Changed the reply, but the assistant didn't list it" in c.value for c in app.caption)


def test_memory_panel_can_restrict_a_memory_without_a_reply(app, tmp_path):
    app.selectbox(key="nft_pick_m1").set_value("summarise").run()
    app.button(key="nft_apply_m1").click().run()
    assert not app.exception
    assert any(c.value == "⛔ Not used for summaries" for c in app.caption)
    options = app.selectbox(key="nft_pick_m1").options
    assert "summaries" not in options and "explanations" in options
    assert "from memory panel" in _all_logs(tmp_path)


def test_check_lets_the_user_see_the_reply_without_a_memory(app):
    app.chat_input[0].set_value(DINNER).run()
    turn = _assistant(app)[0]["turn_id"]
    app.button(key=f"infl_{turn}").click().run()
    assert not app.exception
    results = _assistant(app)[0]["influence"]
    assert all(r.reply_without for r in results)
    without_m3 = next(r for r in results if r.id == "m3").reply_without
    assert any(m.value == without_m3 for m in app.markdown)   # shown inside the popover


def test_forget_also_removes_the_reply_without_it(app):
    app.chat_input[0].set_value(DINNER).run()
    turn = _assistant(app)[0]["turn_id"]
    app.button(key=f"infl_{turn}").click().run()
    next(b for b in app.button if (b.key or "") == f"forget_{turn}:m3").click().run()
    r = next(r for r in _assistant(app)[0]["influence"] if r.id == "m3")
    assert r.reply_without == "[forgotten]"


def test_check_can_be_run_again_on_the_same_reply(app, tmp_path):
    app.chat_input[0].set_value(DINNER).run()
    turn = _assistant(app)[0]["turn_id"]
    app.button(key=f"infl_{turn}").click().run()
    app.button(key=f"infl_again_{turn}").click().run()
    assert "influence" not in _assistant(app)[0]
    app.button(key=f"infl_{turn}").click().run()
    assert not app.exception
    assert _all_logs(tmp_path).count("influence_checked") == 2


# ------------------------------------------------- evaluation mode: reaction, then fit question ---
BIRTHDAY = "Can you write a birthday message for my best friend Sam? Something warm and a bit funny."


def _start_task(app, prompt_start, predict):
    _evaluation(app)
    _button(app, prompt_start).click().run()
    for mid in predict:
        app.checkbox(key=f"pred_{mid}").check()
    if not predict:
        app.checkbox(key="pred_none").check()
    _submit_prediction(app)
    assert not app.exception
    return _assistant(app)[0]


def _log(tmp_path):
    return _all_logs(tmp_path)


def test_new_prompts_wait_until_the_task_is_finished(app):
    _start_task(app, "Can you write a birthday message", ["m2"])
    assert app.chat_input[0].proto.disabled
    assert all(b.disabled for b in app.button if (b.key or "").startswith("demo_"))
    assert any(b.key == "task_done" for b in app.button)
    assert not any("Did it suit your request" in r.label for r in app.radio)   # not asked yet


def test_flagging_a_misapplied_memory_is_appropriate_and_fit_is_scored(app, tmp_path):
    msg = _start_task(app, "Can you write a birthday message", ["m2"])
    app.button(key=f"flag_{msg['turn_id']}").click().run()
    app.button(key="task_done").click().run()
    assert "task_finished" in _log(tmp_path) and "reliance=appropriate" in _log(tmp_path)
    assert "misapplied_shown=m2" in _log(tmp_path)
    app.radio(key="fit_m2").set_value("no")
    next(b for b in app.button if b.label == "Next task →").click().run()
    assert not app.exception
    assert "answer=no;truth=misapplied;correct=True;scenario=S3-birthday" in _log(tmp_path)
    assert app.session_state["eval_task"] is None
    assert not app.chat_input[0].proto.disabled                              # next task can start


def test_accepting_a_misapplied_memory_is_over_reliance(app, tmp_path):
    msg = _start_task(app, "Can you write a birthday message", ["m2"])
    app.button(key=f"ok_{msg['turn_id']}").click().run()
    app.button(key="task_done").click().run()
    assert "accepted=True;flagged=False;corrected=none" in _log(tmp_path)
    assert "reliance=over-reliance" in _log(tmp_path)


def test_correcting_the_memory_counts_and_the_fit_question_is_about_the_original_reply(app, tmp_path):
    msg = _start_task(app, "Can you write a birthday message", [])
    app.button(key=f"nft_{msg['turn_id']}:m2").click().run()                 # rewritten without m2
    assert len(_assistant(app)) == 2
    app.button(key="task_done").click().run()
    assert "corrected=m2" in _log(tmp_path) and "reliance=appropriate" in _log(tmp_path)
    assert any("You want a formal, professional tone" in r.label for r in app.radio)


def test_accepting_fitting_memories_is_appropriate_and_both_are_asked_about(app, tmp_path):
    msg = _start_task(app, "Can you suggest something I could make", ["m3", "m4"])
    app.button(key=f"ok_{msg['turn_id']}").click().run()
    app.button(key="task_done").click().run()
    assert "reliance=appropriate" in _log(tmp_path)
    assert {r.key for r in app.radio if (r.key or "").startswith("fit_")} == {"fit_m3", "fit_m4"}


def test_every_fit_question_must_be_answered(app, tmp_path):
    _start_task(app, "Can you suggest something I could make", ["m3", "m4"])
    app.button(key="task_done").click().run()
    app.radio(key="fit_m3").set_value("yes")
    next(b for b in app.button if b.label == "Next task →").click().run()
    assert any("answer every question" in w.value for w in app.warning)
    assert "fit_judged" not in _log(tmp_path)
    assert app.session_state["eval_task"] is not None


def test_opening_a_source_is_counted_as_checking_effort(app, tmp_path):
    msg = _start_task(app, "Can you write a birthday message", ["m2"])
    app.button(key=f"src_{msg['turn_id']}:m2").click().run()
    app.button(key="task_done").click().run()
    assert "sources_opened=1" in _log(tmp_path)


def test_no_fit_question_when_no_memory_was_used(app, tmp_path):
    app.selectbox(key="nft_pick_m2").set_value("personal_message").run()      # m2 restricted beforehand
    app.button(key="nft_apply_m2").click().run()
    msg = _start_task(app, "Can you write a birthday message", [])
    assert msg["reply"].used == []
    app.button(key="task_done").click().run()
    assert not app.exception
    assert "reliance=n/a" in _log(tmp_path)
    assert app.session_state["eval_task"] is None                          # closed straight away


def test_clear_chat_ends_an_open_task(app):
    _start_task(app, "Can you write a birthday message", ["m2"])
    app.button(key="clear_chat").click().run()
    assert app.session_state["eval_task"] is None and not app.session_state["awaiting_fit"]
    assert not app.chat_input[0].proto.disabled


def test_outside_evaluation_mode_nothing_is_locked(app):
    app.chat_input[0].set_value(BIRTHDAY).run()
    assert not any(b.key == "task_done" for b in app.button)
    assert not app.chat_input[0].proto.disabled


# ------------------------------------------------- evaluation mode: questionnaire ---
def _answer_all(app, score=5):
    from mpa.evaluation import QUESTIONNAIRE
    for item, _, _ in QUESTIONNAIRE:
        app.radio(key=f"q_{item}").set_value(score)


def _finish_one_task(app):
    msg = _start_task(app, "Can you write a birthday message", ["m2"])
    app.button(key=f"flag_{msg['turn_id']}").click().run()
    app.button(key="task_done").click().run()
    app.radio(key="fit_m2").set_value("no")
    next(b for b in app.button if b.label == "Next task →").click().run()


def test_finish_session_appears_only_between_tasks(app):
    msg = _start_task(app, "Can you write a birthday message", ["m2"])
    assert not any(b.key == "finish_session" for b in app.button)          # task still open
    app.button(key="task_done").click().run()
    app.radio(key="fit_m2").set_value("no")
    next(b for b in app.button if b.label == "Next task →").click().run()
    assert any(b.key == "finish_session" for b in app.button)
    assert any("Tasks finished: 1" in c.value for c in app.caption)


def test_questionnaire_is_logged_separately_and_ends_the_session(app, tmp_path):
    _finish_one_task(app)
    app.button(key="finish_session").click().run()
    assert app.chat_input[0].proto.disabled
    _answer_all(app, 6)
    app.text_area(key="q_comment").input("Liked seeing the sources; a bit slow")
    next(b for b in app.button if b.label == "Submit").click().run()
    assert not app.exception
    log = _log(tmp_path)
    assert log.count("questionnaire_answered") == 7                          # 6 items + comment
    assert "item=trust;score=6;reverse=False" in log and "item=effort;score=6;reverse=True" in log
    assert "item=comment;text=Liked seeing the sources, a bit slow" in log
    assert "session_finished" in log and "tasks=1" in log
    assert any("end of the session" in s.value for s in app.success)
    assert app.chat_input[0].proto.disabled                                  # nothing more to do


def test_every_statement_must_be_answered(app, tmp_path):
    _evaluation(app)
    app.button(key="finish_session").click().run()
    app.radio(key="q_trust").set_value(4)
    next(b for b in app.button if b.label == "Submit").click().run()
    assert any("answer every statement" in w.value for w in app.warning)
    assert "questionnaire_answered" not in _all_logs(tmp_path)


def test_new_session_gets_a_new_session_and_fresh_state(app, tmp_path):
    _finish_one_task(app)
    old = app.session_state["session_id"]
    app.radio(key="condition").set_value("B").run()
    app.button(key="new_session").click().run()
    assert not app.exception
    assert app.session_state["session_id"] != old
    assert app.session_state["messages"] == [] and app.session_state["questionnaire"] is None
    rows = [r for r in _log(tmp_path).splitlines() if "session_started" in r]
    assert len(rows) == 1 and app.session_state["session_id"] in rows[0] and "condition=B" in rows[0]


def test_no_questionnaire_outside_evaluation_mode(app):
    app.chat_input[0].set_value(BIRTHDAY).run()
    assert not any(b.key == "finish_session" for b in app.button)


# ------------------------------------------------- locked, isolated sessions ---
def _locked(tmp_path, monkeypatch, condition="B"):
    monkeypatch.setenv("MODE", "replay")
    monkeypatch.setenv("MPA_RUNTIME_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["evaluate"] = "1"
    at.query_params["condition"] = condition
    at.run()
    assert not at.exception
    return at


def test_locked_session_fixes_the_condition_and_hides_developer_controls(tmp_path, monkeypatch):
    at = _locked(tmp_path, monkeypatch, "B")
    assert at.session_state["condition"] == "B" and at.session_state["evaluation_mode"] is True
    keys = {w.key for w in list(at.radio) + list(at.toggle) + list(at.button)}
    assert not keys & {"condition", "developer_view", "evaluation_mode", "influence_check",
                       "reset_demo", "new_session", "clear_chat"}
    assert not any((b.key or "").startswith("src_") for b in at.button)    # no sources in B


def test_locked_condition_is_enforced_on_every_run(tmp_path, monkeypatch):
    at = _locked(tmp_path, monkeypatch, "B")
    at.session_state["condition"] = "A"           # e.g. stale state from before the lock
    at.session_state["developer_view"] = True
    at.run()
    assert at.session_state["condition"] == "B" and at.session_state["developer_view"] is False


def _session(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "replay")
    monkeypatch.setenv("MPA_RUNTIME_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_a_correction_in_one_session_does_not_change_another(tmp_path, monkeypatch):
    first, second = _session(tmp_path, monkeypatch), _session(tmp_path, monkeypatch)
    assert first.session_state["session_id"] != second.session_state["session_id"]
    first.selectbox(key="nft_pick_m2").set_value("personal_message").run()
    first.button(key="nft_apply_m2").click().run()
    second.chat_input[0].set_value(BIRTHDAY).run()
    reply = [m for m in second.session_state["messages"] if m["role"] == "assistant"][0]
    assert {u["id"] for u in reply["reply"].used} == {"m2"}            # still the seeded memory


def test_each_session_writes_only_its_own_log(tmp_path, monkeypatch):
    first, second = _session(tmp_path, monkeypatch), _session(tmp_path, monkeypatch)
    first.chat_input[0].set_value(DINNER).run()
    second.chat_input[0].set_value(BIRTHDAY).run()
    for at in (first, second):
        sid = at.session_state["session_id"]
        rows = (tmp_path / "sessions" / sid / "interaction_log.csv").read_text().splitlines()[1:]
        assert rows and all(f",{sid}," in r for r in rows)


def test_new_session_starts_from_fresh_memories_and_keeps_the_old_files(app, tmp_path):
    old = app.session_state["session_id"]
    app.button(key=f"forget_panel:m1").click().run()
    app.button(key="new_session").click().run()
    new = app.session_state["session_id"]
    assert '"m1"' not in (tmp_path / "sessions" / old / "memories.json").read_text()
    assert '"m1"' in (tmp_path / "sessions" / new / "memories.json").read_text()
