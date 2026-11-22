import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import CASES, score_case  # noqa: E402

from mpa.llm import ReplayModel  # noqa: E402


def _case(case_id):
    return next(c for c in json.loads(CASES.read_text())["cases"] if c["id"] == case_id)


def test_scripted_misapplication_fails_the_over_application_criterion(store):
    row = score_case(_case("E02"), ReplayModel(), store)
    assert row["C1"] and row["C2"] and row["C3"] and not row["C4"]


def test_correction_makes_the_same_case_pass(store):
    store.exclude_for_task("m1", "explain")
    assert score_case(_case("E02"), ReplayModel(), store)["pass"]


def test_new_held_out_cases_exist_and_are_marked():
    cases = {c["id"]: c for c in json.loads(CASES.read_text())["cases"]}
    assert {"E11", "E12", "E13", "E14"} <= set(cases)
    assert all(cases[i]["held_out"] for i in ("E11", "E12", "E13", "E14"))


def test_corrected_condition_withholds_the_scoped_memories():
    from run_eval import run_once
    rows = {r["id"]: r for r in run_once([_case("E03")], ReplayModel(), "corrected", "v2")}
    assert rows["E03"]["pass"]                      # the formal tone is no longer sent for the birthday message
    rows = {r["id"]: r for r in run_once([_case("E03")], ReplayModel(), "default", "v2")}
    assert not rows["E03"]["pass"]


def test_summary_counts_passes_and_reports_across_runs():
    from run_eval import summarise_runs
    run = lambda used, ok: [{"id": "E01", "task": "cooking", "used": used, "pass": ok,
                             "C1": True, "C2": True, "C3": ok, "C4": True}]
    s = summarise_runs([run("m3,m4", True), run("m4", False), run("m3,m4", True)])[0]
    assert s["passed"] == 2 and s["runs"] == 3
    assert s["reported"] == "m3 2/3, m4 3/3" and s["failed"] == "C3"


def test_unknown_prompt_version_is_rejected(store):
    import pytest
    from mpa.context import assemble
    with pytest.raises(ValueError):
        assemble("hi", store.all(), "other", None, prompt_version="v99")
