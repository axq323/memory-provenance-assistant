"""Make eval/sample_session_log_SIMULATED.csv: four SIMULATED evaluation sessions.

No real people took part. The script drives the app itself (Streamlit AppTest,
replay mode) through four complete sessions, two with memory sources shown (A)
and two hidden (B), with scripted reactions that cover the outcomes the
analysis reports: a scoped correction, a Forget (over-correction), accepting a
misapplied memory (over-reliance), flagging a fitting one (under-reliance) and
finishing without a decision. The behaviours and scores are deliberately mixed
across the two settings, so the sample shows no built-in difference between them.
It exists to show the log format and to test eval/summarise_sessions.py end to
end. Its numbers say nothing about users.
Timings are near zero because the clicks are scripted.

Usage (from the repo root):
    python eval/make_sample_log.py                  # rewrites eval/sample_session_log_SIMULATED.csv
    python eval/make_sample_log.py --out other.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval" / "sample_session_log_SIMULATED.csv"

# Per session: memory-sources setting, questionnaire score, and per task
# (prediction, reaction, fit answer, reason). Reactions: ok, flag, nft (Not for this
# kind of request), forget, none (finish without a decision).
SESSIONS = [
    ("A", 6, {"S1-dinner": (["m3", "m4"], "ok", "yes", "Vegetarian and quick, that's what I need."),
              "S3-birthday": (["m2"], "nft", "no", "Formal is right for work, not for a friend."),
              "S4-work-email": (["m2"], "ok", "yes", "It's going to my manager, so formal is fine."),
              "S2-explain": (["m1"], "nft", "no", "I asked for a proper explanation, not a summary.")}),
    ("A", 3, {"S1-dinner": (["m3"], "ok", "unsure", "Looked good."),
              "S3-birthday": (["m2"], "forget", "no", "I don't want it to write formally."),
              "S4-work-email": ([], "ok", None, "Fine."),
              "S2-explain": (["none"], "ok", "yes", "It answered the question, I suppose.")}),
    ("B", 5, {"S1-dinner": (["m3", "m4"], "ok", "yes", "It matched what I asked."),
              "S3-birthday": (["m2"], "ok", "yes", "It was a birthday message, so it did the job."),
              "S4-work-email": (["m2"], "ok", "yes", "Good for work."),
              "S2-explain": (["m1"], "flag", "no", "It felt a bit thin for a step-by-step question.")}),
    ("B", 4, {"S1-dinner": (["m4"], "flag", "unsure", "Not sure why it picked that."),
              "S3-birthday": (["m2"], "nft", "no", "Way too formal for Sam."),
              "S4-work-email": (["m2"], "ok", "yes", "Fine for my manager."),
              "S2-explain": (["m1"], "none", "no", "Didn't really think about it.")}),
]


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def run_session(at, condition: str, score: int, plan: dict) -> None:
    from mpa.evaluation import QUESTIONNAIRE
    at.radio(key="condition").set_value(condition).run()
    at.button(key="new_session").click().run()
    for scenario_id in at.session_state["eval_schedule"]:
        predict, react, fit, reason = plan[scenario_id]
        at.button(key="start_task").click().run()
        for mid in predict:
            at.checkbox(key=f"pred_{mid}" if mid != "none" else "pred_none").check()
        if not predict:
            at.checkbox(key="pred_none").check()
        _click(at, "Show me the reply")
        msg = [m for m in at.session_state["messages"] if m["role"] == "assistant"][-1]
        used = [u["id"] for u in msg["reply"].used]
        turn = msg["turn_id"]
        if react in ("ok", "flag"):
            at.button(key=f"{react}_{turn}").click().run()
        elif react in ("nft", "forget") and used:
            at.button(key=f"{react}_{turn}:{used[0]}").click().run()
        at.button(key="task_done").click().run()
        if at.session_state["eval_task"] is None:          # no memory used: nothing to ask
            continue
        at.text_area(key="reason_text").input(reason)
        _click(at, "Continue →")
        for mid in used:
            at.radio(key=f"fit_{mid}").set_value(fit)
        _click(at, "Next task →")
    at.button(key="finish_session").click().run()
    for item, _, _ in QUESTIONNAIRE:
        at.radio(key=f"q_{item}").set_value(score)
    _click(at, "Submit")
    if at.exception:
        raise RuntimeError(at.exception)


def make(out: Path = OUT) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MODE"] = "replay"
        os.environ["MPA_RUNTIME_DIR"] = tmp
        os.environ.pop("MPA_EVAL_SCHEDULE", None)
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        at.run()
        at.toggle(key="evaluation_mode").set_value(True).run()
        for condition, score, plan in SESSIONS:
            run_session(at, condition, score, plan)
        rows, fields = [], None
        for log in sorted(Path(tmp, "sessions").glob("*/interaction_log.csv")):
            with log.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                rows += [r for r in reader if r["event"] != "demo_reset"]
        # the first (empty) session the app opens with has no evaluation rows; keep only sessions with tasks
        keep = {r["session_id"] for r in rows if r["event"] == "prediction_made"}
        rows = sorted((r for r in rows if r["session_id"] in keep), key=lambda r: (r["session_id"], r["timestamp"]))
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    out = make(Path(parser.parse_args().out))
    print(f"Wrote {out} (SIMULATED: scripted sessions, not real participants)")


if __name__ == "__main__":
    main()
