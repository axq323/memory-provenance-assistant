import pytest

from mpa.interaction_log import log_event, read_log


def test_events_append_with_header_once(tmp_path):
    p = tmp_path / "log.csv"
    common = dict(session_id="p01", condition="A", mode="replay", path=p)
    log_event("reply_shown", turn_id="t1", task="explain", **common)
    log_event("not_for_task", turn_id="t1", task="explain", memory_id="m1", **common)
    rows = read_log(p)
    assert [r["event"] for r in rows] == ["reply_shown", "not_for_task"]
    assert p.read_text().count("timestamp") == 1


def test_unknown_event_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        log_event("clicked_something", session_id="p", condition="A", mode="replay",
                  path=tmp_path / "log.csv")
