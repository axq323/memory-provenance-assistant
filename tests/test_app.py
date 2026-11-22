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
    log = (tmp_path / "interaction_log.csv").read_text()
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
    log = (tmp_path / "interaction_log.csv").read_text()
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
    assert "chat_cleared" in (tmp_path / "interaction_log.csv").read_text()


def test_no_report_caption_does_not_claim_no_memories_were_used(app):
    app.chat_input[0].set_value("Tell me a joke").run()
    captions = [c.value for c in app.caption]
    assert "The assistant didn't report using any memories." in captions
    assert not any("No stored memories were used" in c for c in captions)