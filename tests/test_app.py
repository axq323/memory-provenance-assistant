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
    assert len([b for b in app.button if b.key.startswith("demo_")]) == 3


def test_demo_prompt_shows_reply_with_memory_chips(app):
    _button(app, "I don't really understand how solar").click().run()
    texts = " ".join(m.value for m in app.markdown)
    assert "Memories used in this reply" in texts
    assert "short bullet-point summaries" in texts


def test_show_source_reveals_originating_turn(app):
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if b.key.startswith("src_") and "panel" not in b.key).click().run()
    assert not app.exception
    assert any("saved from this message" in i.value for i in app.info)
