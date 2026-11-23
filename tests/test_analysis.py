"""Evaluation-mode summary: hand-built logs with known answers, plus an end-to-end run through the app."""

import subprocess
import sys
from pathlib import Path

import pytest

from mpa.analysis import (by_scenario, condition_table, load_rows, parse_detail,
                          summarise_conditions, summarise_sessions)
from mpa.interaction_log import log_event

ROOT = Path(__file__).resolve().parent.parent


def _write(path, session, condition, events):
    """events: (event, turn_id, memory_id, detail)"""
    for event, turn, mid, detail in events:
        log_event(event, session_id=session, condition=condition, mode="replay",
                  turn_id=turn, memory_id=mid, detail=detail, path=path)


def _questionnaire(score, effort):
    rows = [("questionnaire_answered", "", "", f"item={i};score={score};reverse=False")
            for i in ("trust", "understanding", "predictability", "control", "comfort")]
    rows.append(("questionnaire_answered", "", "", f"item=effort;score={effort};reverse=True"))
    return rows + [("session_finished", "", "", "tasks=2")]


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "log.csv"
    # Session p1 (sources shown): birthday -> flagged (appropriate), fit correct;
    #                     dinner -> accepted (appropriate), one fit right, one "not sure"
    _write(path, "p1", "A", [
        ("prediction_made", "t1", "", "predicted=m2;used=m2;exact=True;jaccard=1.0;scenario=S3-birthday"),
        ("source_opened", "t1", "m2", ""),
        ("reply_flagged", "t1", "", "used=m2"),
        ("task_finished", "t1", "", "scenario=S3-birthday;accepted=False;flagged=True;corrected=m2;"
                                    "sources_opened=1;seconds=60;misapplied_shown=m2;reliance=appropriate;"
                                    "correction=scoped"),
        ("fit_judged", "t1", "m2", "answer=no;truth=misapplied;correct=True;scenario=S3-birthday"),
        ("prediction_made", "t2", "", "predicted=m3;used=m3,m4;exact=False;jaccard=0.5;scenario=S1-dinner"),
        ("reply_accepted", "t2", "", "used=m3,m4"),
        ("task_finished", "t2", "", "scenario=S1-dinner;accepted=True;flagged=False;corrected=none;"
                                    "sources_opened=0;seconds=30;misapplied_shown=none;reliance=appropriate"),
        ("fit_judged", "t2", "m3", "answer=yes;truth=fits;correct=True;scenario=S1-dinner"),
        ("fit_judged", "t2", "m4", "answer=unsure;truth=fits;correct=False;scenario=S1-dinner"),
    ] + _questionnaire(6, 2))
    # Session p2 (sources hidden): birthday -> accepted (over-reliance), fit wrong
    _write(path, "p2", "B", [
        ("prediction_made", "t3", "", "predicted=none;used=m2;exact=False;jaccard=0.0;scenario=S3-birthday"),
        ("reply_accepted", "t3", "", "used=m2"),
        ("task_finished", "t3", "", "scenario=S3-birthday;accepted=True;flagged=False;corrected=none;"
                                    "sources_opened=0;seconds=20;misapplied_shown=m2;reliance=over-reliance"),
        ("fit_judged", "t3", "m2", "answer=yes;truth=misapplied;correct=False;scenario=S3-birthday"),
    ] + _questionnaire(4, 5))
    # Developer session without predictions (live testing): must be ignored
    _write(path, "dev", "A", [("reply_shown", "t9", "", "used=m2"), ("influence_checked", "t9", "", "m2=agrees")])
    # Incomplete session p3 (sources shown): one task, no questionnaire
    _write(path, "p3", "A", [
        ("prediction_made", "t4", "", "predicted=m1;used=m1;exact=True;jaccard=1.0;scenario=S2-explain"),
        ("task_finished", "t4", "", "scenario=S2-explain;accepted=False;flagged=False;corrected=m1;"
                                    "sources_opened=2;seconds=90;misapplied_shown=m1;reliance=appropriate;"
                                    "correction=forgot"),
    ])
    return path


def _people(log):
    return {p["session"]: p for p in summarise_sessions(load_rows([log]))}


def test_parse_detail_keeps_equals_signs_in_values():
    assert parse_detail("item=comment;text=a=b, c") == {"item": "comment", "text": "a=b, c"}
    assert parse_detail("") == {}


def test_only_study_sessions_are_analysed(log):
    people = _people(log)
    assert set(people) == {"p1", "p2", "p3"}
    assert people["p1"]["complete"] and not people["p3"]["complete"]


def test_session_outcomes(log):
    p1, p2 = _people(log)["p1"], _people(log)["p2"]
    assert p1["condition"] == "A" and p1["tasks"] == 2
    assert p1["prediction_exact"] == 0.5 and p1["prediction_overlap"] == 0.75
    assert p1["fit_accuracy"] == pytest.approx(2 / 3, abs=0.001)     # "not sure" counts as not correct
    assert p1["fit_unsure"] == pytest.approx(1 / 3, abs=0.001)
    assert p1["reliance_appropriate"] == 1.0
    assert p1["over_reliance"] == 0.0 and p1["under_reliance"] == 0.0
    assert p1["sources_per_task"] == 0.5 and p1["seconds_per_task"] == 45
    assert p1["q_trust"] == 6 and p1["q_effort"] == 6                 # effort 2 reversed -> 6
    assert p2["over_reliance"] == 1.0 and p2["under_reliance"] is None  # no suitable-reply task
    assert p2["fit_accuracy"] == 0.0 and p2["q_effort"] == 3
    # diagnosis and correction are separate outcomes
    assert p1["scope_error_identified"] == 1.0 and p1["correction_scoped"] == 1.0
    assert p2["scope_error_identified"] == 0.0 and p2["correction_scoped"] is None   # older log rows
    p3 = _people(log)["p3"]
    assert p3["correction_forgot"] == 1.0 and p3["correction_scoped"] == 0.0


def test_condition_summary_excludes_incomplete_by_default(log):
    people = summarise_sessions(load_rows([log]))
    summary = summarise_conditions(people)
    assert summary["A"]["n"] == 1 and summary["B"]["n"] == 1          # p3 left out
    assert summary["A"]["reliance_appropriate"]["mean"] == 1.0
    assert summarise_conditions(people, complete_only=False)["A"]["n"] == 2


def test_table_shows_the_a_minus_b_difference(log):
    table = condition_table(summarise_conditions(summarise_sessions(load_rows([log]))))
    assert table[0][-1] == "Shown - hidden"
    assert table[0][1].startswith("Sources shown")
    over = next(r for r in table if r[0].startswith("Over-reliance"))
    assert over[-1] == "-1.00"                                        # A 0.00 vs B 1.00


def test_scenario_breakdown(log):
    rows = by_scenario(load_rows([log]), sessions={"p1", "p2"})
    cell = next(r for r in rows if r["condition"] == "B" and r["scenario"] == "S3-birthday")
    assert cell == {"condition": "B", "scenario": "S3-birthday", "tasks": 1, "prediction_exact": 0.0,
                    "fit_accuracy": 0.0, "reliance_appropriate": 0.0}


def test_the_same_log_passed_twice_is_not_double_counted(log):
    assert len(load_rows([log, log])) == len(load_rows([log]))


def test_script_runs_and_writes_results(log, tmp_path):
    out = tmp_path / "results"
    res = subprocess.run([sys.executable, str(ROOT / "eval" / "summarise_sessions.py"), str(log),
                          "--out", str(out), "--note", "test data"],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert "evaluation sessions: 3 (2 complete) · summarised: 2" in res.stdout
    assert {p.name.split("_", 1)[1] for p in out.iterdir()} == {"sessions.csv", "scenarios.csv",
                                                                 "summary.md"}
    assert "| Measure |" in next(out.glob("*_summary.md")).read_text(encoding="utf-8")


# ------------------------------------------------ end to end through the app ---
def test_two_sessions_through_the_app_are_summarised(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("MODE", "replay")
    monkeypatch.setenv("MPA_RUNTIME_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.run()
    app.toggle(key="evaluation_mode").set_value(True).run()

    def session(condition, react):
        app.radio(key="condition").set_value(condition).run()
        app.button(key="new_session").click().run()
        next(b for b in app.button if b.label.startswith("Can you write a birthday message")).click().run()
        app.checkbox(key="pred_m2").check()
        next(b for b in app.button if b.label == "Show me the reply").click().run()
        msg = [m for m in app.session_state["messages"] if m["role"] == "assistant"][0]
        app.button(key=f"{react}_{msg['turn_id']}").click().run()
        app.button(key="task_done").click().run()
        app.text_area(key="reason_text").input("It sounded wrong for a friend" if react == "flag" else "Fine")
        next(b for b in app.button if b.label == "Continue →").click().run()
        app.radio(key="fit_m2").set_value("no" if react == "flag" else "yes")
        next(b for b in app.button if b.label == "Next task →").click().run()
        app.button(key="finish_session").click().run()
        for item in ("trust", "understanding", "predictability", "control", "effort", "comfort"):
            app.radio(key=f"q_{item}").set_value(5)
        next(b for b in app.button if b.label == "Submit").click().run()
        assert not app.exception

    session("A", "flag")     # questions the misapplied formal tone
    session("B", "ok")       # accepts it
    logs = sorted((tmp_path / "sessions").glob("*/interaction_log.csv"))
    people = {p["condition"]: p for p in summarise_sessions(load_rows(logs))}
    assert set(people) == {"A", "B"} and all(p["complete"] for p in people.values())
    assert people["A"]["reliance_appropriate"] == 1.0 and people["A"]["fit_accuracy"] == 1.0
    assert people["B"]["over_reliance"] == 1.0 and people["B"]["fit_accuracy"] == 0.0
    assert people["A"]["prediction_exact"] == people["B"]["prediction_exact"] == 1.0


def test_the_simulated_sample_log_analyses(tmp_path):
    sample = ROOT / "eval" / "sample_session_log_SIMULATED.csv"
    people = summarise_sessions(load_rows([sample]))
    assert len(people) == 4 and {p["condition"] for p in people} == {"A", "B"}
    assert all(p["mode"] == "replay" and p["complete"] for p in people)
