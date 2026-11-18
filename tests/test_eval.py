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
