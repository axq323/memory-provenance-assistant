import csv
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


# ------------------------------------------------- what the criteria can and can't see ---
class _Scripted:
    """A model that always returns one fixed reply."""
    name = "scripted"

    def __init__(self, answer, used):
        self.raw = json.dumps({"answer": answer, "memories_used": [{"id": m, "how": ""} for m in used]})

    def generate(self, record, user_text, temperature=None):
        return self.raw


def test_results_keep_the_reply_text(store):
    assert "Dear Sam" in score_case(_case("E03"), ReplayModel(), store)["answer"]


def test_blind_spot_a_non_answer_that_lists_the_right_memories_passes(store):
    # why a manual check is needed: C1-C4 only see the report
    row = score_case(_case("E01"), _Scripted("I do not know.", ["m3", "m4"]), store)
    assert row["pass"]


def test_blind_spot_a_misapplied_reply_that_reports_nothing_passes(store):
    formal = "Dear Sam, I would like to extend my warmest wishes on the occasion of your birthday."
    assert score_case(_case("E03"), _Scripted(formal, []), store)["pass"]


def test_new_runs_are_logged_above_the_manual_checks_section(tmp_path):
    from run_eval import append_log_row
    log = tmp_path / "prompt_log.md"
    log.write_text("| Run |\n|---|\n| old |\n\n## Manual checks\n\n| Sheet |\n|---|\n| s1 |\n")
    append_log_row("| new |", log)
    text = log.read_text()
    assert text.index("| new |") < text.index("## Manual checks") and text.index("| old |") < text.index("| new |")
    assert text.rstrip().endswith("| s1 |")


def _results(tmp_path, with_answer=True):
    path = tmp_path / "20261123-000000_live_v4_default.csv"
    fields = ["run", "prompt", "condition", "id", "task", "used", "dropped", "C1", "C2", "C3", "C4",
              "pass", "error"] + (["answer"] if with_answer else [])
    rows = [
        {"run": 1, "id": "E01", "used": "m3,m4", "pass": True, "answer": "I do not know."},
        {"run": 1, "id": "E03", "used": "", "pass": True, "answer": "Dear Sam, I would like to extend..."},
        {"run": 1, "id": "E04", "used": "m2", "pass": True, "answer": "Dear Ms Patel, ..."},
        {"run": 2, "id": "E01", "used": "m3", "pass": False, "answer": "Chickpea curry, 25 min."},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    return path


def test_manual_check_sheet_and_score(tmp_path):
    from manual_check import make_sheet, score_sheet
    sheet = make_sheet(_results(tmp_path), tmp_path / "checks")
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["case"] for r in rows] == ["E01", "E03", "E04"]              # first run only
    assert rows[0]["should_use"] == "m3,m4" and rows[0]["answer"] == "I do not know."
    verdicts = {"E01": ("n", "na", "y"), "E03": ("y", "na", "n"), "E04": ("y", "y", "y")}
    for r in rows:
        r["answers_request"], r["follows_fitting"], r["avoids_misapplied"] = verdicts[r["case"]]
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = score_sheet(sheet)
    assert result["checked"] == 3 and result["manual_pass"] == 1 and result["agree"] == 1
    assert {case for _, case, _, _ in result["disagreements"]} == {"E01", "E03"}


def test_manual_check_needs_a_run_that_kept_the_replies(tmp_path):
    import pytest
    from manual_check import make_sheet
    with pytest.raises(SystemExit, match="no 'answer' column"):
        make_sheet(_results(tmp_path, with_answer=False), tmp_path / "checks")


def test_manual_check_script_runs(tmp_path):
    import subprocess
    res = subprocess.run([sys.executable, str(ROOT / "eval" / "manual_check.py"), "make",
                          str(_results(tmp_path)), "--out", str(tmp_path / "checks")],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "checks" / "20261123-000000_live_v4_default_sheet.csv").exists()
