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
