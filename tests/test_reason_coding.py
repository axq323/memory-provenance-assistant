"""Diagnosis is two measures: the fit answer, and whether the written reason names the scope error."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mpa.analysis import load_rows, summarise_sessions
from mpa.reason_coding import load_codes, rubric_markdown, write_sheet

ROOT = Path(__file__).resolve().parent.parent


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run()


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """One session: dinner (accepted), then the birthday message flagged with the reason
    'Its tone is too formal' and the fit answer No."""
    monkeypatch.setenv("MODE", "replay")
    monkeypatch.setenv("MPA_RUNTIME_DIR", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    at.toggle(key="evaluation_mode").set_value(True).run()
    at.session_state["eval_schedule"] = ["S1-dinner", "S3-birthday"]
    at.run()
    for mids, react, reason, fit in ((["m3", "m4"], "ok", "Quick and vegetarian.", "yes"),
                                     (["m2"], "flag", "Its tone is too formal", "no")):
        at.button(key="start_task").click().run()
        for mid in mids:
            at.checkbox(key=f"pred_{mid}").check()
        _click(at, "Show me the reply")
        msg = [m for m in at.session_state["messages"] if m["role"] == "assistant"][-1]
        at.button(key=f"{react}_{msg['turn_id']}").click().run()
        at.button(key="task_done").click().run()
        at.text_area(key="reason_text").input(reason)
        _click(at, "Continue →")
        for mid in mids:
            at.radio(key=f"fit_{mid}").set_value(fit)
        _click(at, "Next task →")
    assert not at.exception
    return sorted((tmp_path / "sessions").glob("*/interaction_log.csv"))


def _code(sheet, code):
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["code"] = code
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_the_fit_answer_alone_is_reported_as_judged_unsuitable_not_as_understood(logs):
    [s] = summarise_sessions(load_rows(logs))
    assert s["misapplied_judged_unsuitable"] == 1.0        # "No" on the formal-tone memory
    assert s["scope_named"] is None                         # not measured until the reasons are coded
    assert "scope_error_identified" not in s


def test_the_sheet_lists_only_misapplied_tasks_and_hides_the_setting(logs, tmp_path):
    sheet = tmp_path / "sheet.csv"
    assert write_sheet(load_rows(logs), sheet) == 1
    with sheet.open(newline="", encoding="utf-8") as f:
        [row] = list(csv.DictReader(f))
    assert "condition" not in row
    assert row["scenario"] == "S3-birthday" and row["reason"] == "Its tone is too formal"
    assert row["memory"].startswith("m2: You want a formal") and "work" in row["what_scope_means"]


def test_too_formal_is_coded_as_mismatch_and_does_not_count_as_naming_the_scope(logs, tmp_path):
    sheet = tmp_path / "sheet.csv"
    write_sheet(load_rows(logs), sheet)
    _code(sheet, "mismatch")
    codes, problems = load_codes(sheet)
    assert not problems
    [s] = summarise_sessions(load_rows(logs), codes)
    assert s["misapplied_judged_unsuitable"] == 1.0 and s["scope_named"] == 0.0
    _code(sheet, "scope")                                   # e.g. "the formal preference was for my manager"
    [s] = summarise_sessions(load_rows(logs), load_codes(sheet)[0])
    assert s["scope_named"] == 1.0


def test_blank_or_unknown_codes_are_reported(logs, tmp_path):
    sheet = tmp_path / "sheet.csv"
    write_sheet(load_rows(logs), sheet)
    assert load_codes(sheet)[1] == ["item 1: not coded"]
    _code(sheet, "maybe")
    assert "unknown code 'maybe'" in load_codes(sheet)[1][0]


def test_the_rubric_gives_examples_of_each_code():
    text = rubric_markdown()
    assert "Its tone is too formal" in text and "emailing my manager" in text and "## S2-explain" in text


def test_the_scripts_make_check_and_summarise(logs, tmp_path):
    sheet = tmp_path / "sheet.csv"
    run = lambda *a: subprocess.run([sys.executable, *a], capture_output=True, text=True, cwd=ROOT)  # noqa: E731
    res = run(str(ROOT / "eval" / "code_reasons.py"), "make", *map(str, logs), "--out", str(sheet))
    assert res.returncode == 0, res.stderr
    assert "1 reasons to code" in res.stdout
    assert run(str(ROOT / "eval" / "code_reasons.py"), "check", str(sheet)).returncode == 1   # not coded yet
    _code(sheet, "mismatch")
    assert run(str(ROOT / "eval" / "code_reasons.py"), "check", str(sheet)).returncode == 0
    res = run(str(ROOT / "eval" / "summarise_sessions.py"), *map(str, logs), "--include-incomplete",
              "--codes", str(sheet), "--out", str(tmp_path / "out"))
    assert res.returncode == 0, res.stderr
    assert "Diagnosis: named the scope error (coded reasons)" in res.stdout
