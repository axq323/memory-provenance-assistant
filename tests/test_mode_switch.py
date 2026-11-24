"""Switching between replay and live mode from the developer controls.

Live mode is never actually called here: whether the endpoint answers is faked with
monkeypatch, so these tests need no model and make no network calls.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mpa.llm

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


def _fake_endpoint(monkeypatch, app, ok: bool):
    """Make live_available() answer ok / not ok, then rerun so the app uses the fake."""
    message = "Live mode on: replies now come from qwen2.5:7b." if ok else "Couldn't reach the live model."
    monkeypatch.setattr(mpa.llm, "live_available", lambda timeout=3.0: (ok, message))
    app.run()


def _logs(tmp_path):
    return "".join(p.read_text() for p in sorted((tmp_path / "sessions").glob("*/interaction_log.csv")))


def _memory(tmp_path, app, memory_id):
    path = tmp_path / "sessions" / app.session_state["session_id"] / "memories.json"
    return next(m for m in json.loads(path.read_text()) if m["id"] == memory_id)


def test_starts_in_the_mode_from_env_and_labels_replies(app):
    assert app.session_state["mode"] == "replay"
    assert app.radio(key="mode_choice").value == "replay"
    app.chat_input[0].set_value(EXPLAIN).run()
    assert any(c.value == "Scripted reply (replay mode)" for c in app.caption)


def test_live_is_refused_when_the_endpoint_does_not_answer(app, tmp_path, monkeypatch):
    _fake_endpoint(monkeypatch, app, ok=False)
    app.chat_input[0].set_value(EXPLAIN).run()
    app.radio(key="mode_choice").set_value("live").run()
    assert not app.exception
    assert app.session_state["mode"] == "replay"
    assert app.session_state["mode_choice"] == "replay"      # the switch is put back
    assert len(app.session_state["messages"]) == 2           # the chat is untouched
    assert "mode_switched" not in _logs(tmp_path)


def test_switching_clears_the_chat_but_keeps_memories_and_corrections(app, tmp_path, monkeypatch):
    _fake_endpoint(monkeypatch, app, ok=True)
    app.chat_input[0].set_value(EXPLAIN).run()
    next(b for b in app.button if (b.key or "").startswith("nft_") and (b.key or "").endswith(":m1")).click().run()
    assert "explain" in _memory(tmp_path, app, "m1")["excluded_tasks"]

    app.radio(key="mode_choice").set_value("live").run()
    assert not app.exception
    assert app.session_state["mode"] == "live"
    assert app.session_state["messages"] == []
    assert any("Mode: **live**" in c.value for c in app.caption)
    assert any("Live mode:" in i.value for i in app.info)
    assert "explain" in _memory(tmp_path, app, "m1")["excluded_tasks"]   # correction kept
    assert "mode_switched" in _logs(tmp_path) and "to=live" in _logs(tmp_path)

    app.radio(key="mode_choice").set_value("replay").run()            # and back
    assert not app.exception
    assert app.session_state["mode"] == "replay"
    assert any("Replay mode" in i.value for i in app.info)
    assert "to=replay" in _logs(tmp_path)


def test_events_after_a_switch_are_logged_with_the_new_mode(app, tmp_path, monkeypatch):
    _fake_endpoint(monkeypatch, app, ok=True)
    app.radio(key="mode_choice").set_value("live").run()
    row = [line for line in _logs(tmp_path).splitlines() if "mode_switched" in line][0]
    assert ",live," in row


def test_the_mode_is_fixed_while_evaluation_mode_is_on(app):
    app.toggle(key="evaluation_mode").set_value(True).run()
    assert app.radio(key="mode_choice").disabled


def test_a_locked_session_has_no_mode_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("MODE", "replay")
    monkeypatch.setenv("MPA_RUNTIME_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["evaluate"] = "1"
    at.run()
    assert not at.exception
    assert not any(r.key == "mode_choice" for r in at.radio)
    assert at.session_state["mode"] == "replay"


def test_live_available_reports_an_unreachable_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")      # nothing listens here
    ok, message = mpa.llm.live_available(timeout=1.0)
    assert ok is False and "Couldn't reach" in message


def test_get_model_and_judge_follow_the_requested_mode(monkeypatch):
    monkeypatch.setenv("MODE", "live")
    assert isinstance(mpa.llm.get_model("replay"), mpa.llm.ReplayModel)
    assert mpa.llm.get_judge("replay") is None
    monkeypatch.setenv("MODE", "replay")
    assert mpa.llm.default_mode() == "replay"
    assert mpa.llm.get_model("live").model == mpa.llm.live_model_name() # type: ignore